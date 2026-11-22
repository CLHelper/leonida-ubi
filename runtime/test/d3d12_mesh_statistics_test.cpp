// D3D-132: D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1 (mesh and amplification counts) on the Vulkan layer, against
// the host D3D12 runtime on the GPU (docs/d3d12_vulkan.md section 4.6).
//   - Heaps: refused with E_NOTIMPL, and OPTIONS9.MeshShaderPipelineStatsSupported FALSE, on a device without
//     meshShaderQueries; created, and reported, with it (as every host adapter tried so far does).
//   - One list, run on both: a VS draw inside a PIPELINE_STATISTICS1 query and the same draw inside a
//     PIPELINE_STATISTICS query (the 11 classic counters must agree, AS/MS/MSPrimitives are 0); DispatchMesh(2, 2, 1)
//     and DispatchMesh(1, 1, 2) in queries of their own; a query never ended (zeros); and a query split by a
//     ResolveQueryData inside it (D3D-162: summed segments), around DispatchMesh(3, 1, 1) and (1, 1, 1). The results
//     are resolved at a non-zero offset over a 0xcd fill. ASInvocations, MSInvocations and MSPrimitives must equal
//     the host's; every counter of both is printed.
// The layer runs test/glsl/ms_draw.mesh, vs_fullscreen.vert and ps_draw.frag; the host runs the HLSL equivalents
// below, compiled to DXIL at run time by dxcompiler.dll (PATH or the Windows SDK). Without it, or without a host
// device that can run mesh shaders, only the layer's own checks run. There are no amplification shaders (the layer
// refuses them, D3D-29), so ASInvocations is 0 on both. No game data. LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <dxcapi.h>
#include <dxgi1_6.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#if !defined(LEONIDA_TEST_MS_SPV) || !defined(LEONIDA_TEST_PS_SPV) || !defined(LEONIDA_TEST_VS_FULLSCREEN_SPV)
#error "LEONIDA_TEST_MS_SPV, LEONIDA_TEST_PS_SPV and LEONIDA_TEST_VS_FULLSCREEN_SPV must name the compiled test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)
#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

constexpr UINT kWords = 14;                    // D3D12_QUERY_DATA_PIPELINE_STATISTICS1
constexpr UINT kAS = 11, kMS = 12, kMSPrimitives = 13;
constexpr UINT64 kStatistics1At = 8;           // queries 0-3 of the PIPELINE_STATISTICS1 heap
constexpr UINT64 kStatisticsAt = 512;          // query 0 of the PIPELINE_STATISTICS heap
constexpr UINT64 kSplitAt = 1024;              // PIPELINE_STATISTICS1 query 4, split by the resolve inside it
constexpr UINT64 kInnerAt = 2048;              // that inner resolve: query 1 again
constexpr UINT kReadbackBytes = 4096;
const char* const kCounterNames[kWords] = {"IAVertices", "IAPrimitives", "VSInvocations", "GSInvocations",
                                           "GSPrimitives", "CInvocations", "CPrimitives", "PSInvocations",
                                           "HSInvocations", "DSInvocations", "CSInvocations", "ASInvocations",
                                           "MSInvocations", "MSPrimitives"};

// The host's shaders: ms_draw.mesh, vs_fullscreen.vert and ps_draw.frag in HLSL, with s0:s1 as b0 and the tint CBV
// as b1 (the root signature below).
const char kHlsl[] = R"(
cbuffer Offset : register(b0) { float2 offset; };
cbuffer Tint : register(b1) { float3 tint; };
struct Vertex { float4 position : SV_Position; float4 color : COLOR0; };
[outputtopology("triangle")] [numthreads(32, 1, 1)]
void ms(uint i : SV_GroupIndex, uint3 group : SV_GroupID, out vertices Vertex verts[4], out indices uint3 tris[2]) {
    SetMeshOutputCounts(4, 2);
    const float column = float(group.x + group.z), row = float(group.y);
    if (i < 4) {
        const float2 corner = float2(float(i & 1u), float(i >> 1));
        const float2 top_left = float2(-1.0 + 0.5 * column, 1.0 - 0.5 * row);
        verts[i].position = float4(top_left + float2(0.5, -0.5) * corner + offset, 0.0, 1.0);
        verts[i].color = float4(1.0, 0.0, 0.0, 1.0);
    }
    if (i == 0) tris[0] = uint3(0, 1, 2);
    if (i == 1) tris[1] = uint3(2, 1, 3);
}
Vertex vs(uint id : SV_VertexID) {
    const float2 corner = float2(float(id & 2u), float((id << 1) & 2u));
    Vertex v;
    v.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    v.color = float4(1.0, 1.0, 0.0, 1.0);
    return v;
}
float4 ps(Vertex v) : SV_Target { return float4(v.color.r * tint.x, tint.y, v.color.b * tint.z, 1.0); }
)";

// One pipeline state stream subobject as d3dx12.h declares it (CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT).
template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type, typename Inner>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = Type;
    Inner inner;
};

template <typename... S>
HRESULT create_pso(ID3D12Device* device, Ref<ID3D12PipelineState>& out, const S&... subobjects) {
    std::vector<std::uint8_t> bytes;
    const auto add = [&](const auto& subobject) {
        const std::size_t at = bytes.size();
        bytes.resize(at + sizeof(subobject));
        std::memcpy(bytes.data() + at, &subobject, sizeof(subobject));
    };
    (add(subobjects), ...);
    Ref<ID3D12Device2> device2;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device2.p)))) return E_NOINTERFACE;
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{bytes.size(), bytes.data()};
    return device2->CreatePipelineState(&desc, IID_PPV_ARGS(&out.p));
}

struct Bytecode {
    D3D12_SHADER_BYTECODE ms, vs, ps;
};

// The root signature and both PSOs (MS + PS, VS + PS).
struct Pipelines {
    Ref<ID3D12RootSignature> root;
    Ref<ID3D12PipelineState> mesh, vs;
};

bool make_pipelines(ID3D12Device* device, const Bytecode& code, Pipelines& out) {
    // [0] two constants (s0, s1: offset), [1] a root CBV (s2:s3: tint), as in d3d12_mesh_test.
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 2};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[1].Descriptor = {1, 0};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(root_desc);
    CHECK(!blob.empty());
    CHECK(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&out.root.p))));
    D3D12_RT_FORMAT_ARRAY targets{};
    targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    targets.NumRenderTargets = 1;
    using Root = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>;
    using Shader = D3D12_SHADER_BYTECODE;
    using Targets = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY>;
    const Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, Shader> ps{.inner = code.ps};
    CHECK(SUCCEEDED(create_pso(device, out.mesh, Root{.inner = out.root.p},
                               Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, Shader>{.inner = code.ms}, ps,
                               Targets{.inner = targets})));
    CHECK(SUCCEEDED(create_pso(
        device, out.vs, Root{.inner = out.root.p}, Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, Shader>{.inner = code.vs},
        ps, Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE>{
                .inner = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE},
        Targets{.inner = targets})));
    return true;
}

// Every resolved UINT64 of one run, from the readback buffer (0xcd where nothing was written).
struct Results {
    std::array<std::array<UINT64, kWords>, 4> statistics1{};  // queries 0-3
    std::array<UINT64, 11> statistics{};
    std::array<UINT64, kWords> split{}, inner{};
    UINT64 past_end = 0;  // the word after query 3
};

UINT64 word(const std::uint8_t* bytes, UINT64 offset) {
    UINT64 value;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

// Records the list described at the top, runs it on `device` and reads the results back. `rtv` names the render
// target view of `target` (64x64 R8G8B8A8, COMMON).
bool run(ID3D12Device* device, const Pipelines& pipelines, ID3D12Resource* target, D3D12_CPU_DESCRIPTOR_HANDLE rtv,
         Results& out) {
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD}, readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> upload, readback;
    const auto upload_desc = buffer_desc(256), readback_desc = buffer_desc(kReadbackBytes);
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                    IID_PPV_ARGS(&upload.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&readback.p))));
    void* mapped = nullptr;
    CHECK(SUCCEEDED(upload->Map(0, nullptr, &mapped)));
    const float tint[3] = {1, 1, 1};
    std::memcpy(mapped, tint, sizeof(tint));
    CHECK(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    std::memset(mapped, 0xcd, kReadbackBytes);
    readback->Unmap(0, nullptr);

    Ref<ID3D12QueryHeap> statistics1, statistics;
    D3D12_QUERY_HEAP_DESC heap_desc{D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1, 5, 0};
    CHECK(SUCCEEDED(device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&statistics1.p))));
    heap_desc = {D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS, 1, 0};
    CHECK(SUCCEEDED(device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&statistics.p))));

    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList6> list6;
    Ref<ID3D12Fence> fence;
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pipelines.vs.p,
                                              IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list6.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    const D3D12_RESOURCE_BARRIER to_target =
        transition(target, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
    list->ResourceBarrier(1, &to_target);
    const FLOAT blue[4] = {0, 0, 1, 1};
    list->ClearRenderTargetView(rtv, blue, 0, nullptr);
    list->SetGraphicsRootSignature(pipelines.root.p);
    const D3D12_VIEWPORT viewport{0, 0, 64, 64, 0, 1};
    const D3D12_RECT scissor{0, 0, 64, 64};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const float origin[2] = {0, 0};
    list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
    list->SetGraphicsRootConstantBufferView(1, upload->GetGPUVirtualAddress());
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    // The same full-screen triangle in each kind of statistics query.
    list->BeginQuery(statistics1.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, 0);
    list->DrawInstanced(3, 1, 0, 0);
    list->EndQuery(statistics1.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, 0);
    list->BeginQuery(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
    list->DrawInstanced(3, 1, 0, 0);
    list->EndQuery(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
    // Mesh draws: 4 and 2 groups of one quad each (query 3 is never begun).
    list->SetPipelineState(pipelines.mesh.p);
    list->BeginQuery(statistics1.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, 1);
    list6->DispatchMesh(2, 2, 1);
    list->EndQuery(statistics1.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, 1);
    list->BeginQuery(statistics1.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, 2);
    list6->DispatchMesh(1, 1, 2);
    list->EndQuery(statistics1.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, 2);
    // A query with a resolve inside it, which the layer splits into two summed segments: 3 + 1 groups.
    list->BeginQuery(statistics1.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, 4);
    list6->DispatchMesh(3, 1, 1);
    list->ResolveQueryData(statistics1.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, 1, 1, readback.p, kInnerAt);
    list6->DispatchMesh(1, 1, 1);
    list->EndQuery(statistics1.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, 4);
    list->ResolveQueryData(statistics1.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, 0, 4, readback.p, kStatistics1At);
    list->ResolveQueryData(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, 1, readback.p, kStatisticsAt);
    list->ResolveQueryData(statistics1.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS1, 4, 1, readback.p, kSplitAt);
    const D3D12_RESOURCE_BARRIER to_common =
        transition(target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
    list->ResourceBarrier(1, &to_common);
    CHECK(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

    CHECK(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    for (UINT q = 0; q < 4; ++q)
        for (UINT i = 0; i < kWords; ++i) out.statistics1[q][i] = word(bytes, kStatistics1At + (q * kWords + i) * 8);
    for (UINT i = 0; i < 11; ++i) out.statistics[i] = word(bytes, kStatisticsAt + i * 8);
    for (UINT i = 0; i < kWords; ++i) {
        out.split[i] = word(bytes, kSplitAt + i * 8);
        out.inner[i] = word(bytes, kInnerAt + i * 8);
    }
    out.past_end = word(bytes, kStatistics1At + 4 * kWords * 8);
    readback->Unmap(0, nullptr);
    return true;
}

// dxcompiler.dll from PATH, else the newest Windows SDK's (with dxil.dll beside it, which signs the DXIL).
HMODULE load_dxcompiler() {
    if (HMODULE m = LoadLibraryW(L"dxcompiler.dll")) return m;
    char root[MAX_PATH] = {};
    std::size_t length = 0;
    if (getenv_s(&length, root, "ProgramFiles(x86)") || !length) return nullptr;
    std::filesystem::path best;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::path(root) / "Windows Kits/10/bin",
                                                                 error)) {
        const auto dll = entry.path() / "x64" / "dxcompiler.dll";
        if (std::filesystem::exists(dll) && std::filesystem::exists(entry.path() / "x64" / "dxil.dll") &&
            (best.empty() || dll > best))
            best = dll;
    }
    return best.empty() ? nullptr : LoadLibraryExW(best.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}

std::vector<std::uint8_t> compile(IDxcCompiler3* compiler, const wchar_t* entry, const wchar_t* profile) {
    const DxcBuffer source{kHlsl, sizeof(kHlsl) - 1, DXC_CP_UTF8};
    LPCWSTR args[] = {L"-E", entry, L"-T", profile};
    Ref<IDxcResult> result;
    HRESULT status = E_FAIL;
    if (FAILED(compiler->Compile(&source, args, 4, nullptr, IID_PPV_ARGS(&result.p))) ||
        FAILED(result->GetStatus(&status)) || FAILED(status)) {
        Ref<IDxcBlobUtf8> errors;
        if (result.p) result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors.p), nullptr);
        std::fprintf(stderr, "dxc %ls: %s\n", entry, errors.p ? errors->GetStringPointer() : "failed");
        return {};
    }
    Ref<IDxcBlob> object;
    if (FAILED(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object.p), nullptr)) || !object.p) return {};
    const auto* data = static_cast<const std::uint8_t*>(object->GetBufferPointer());
    return {data, data + object->GetBufferSize()};
}

// The host's results, or false when it cannot run the list (no dxcompiler, no device, no mesh statistics).
bool run_host(Results& out) {
    if (hostwine::host_is_wine()) return false;
    HMODULE dxc = load_dxcompiler();
    const auto create = dxc ? reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(dxc, "DxcCreateInstance")) : nullptr;
    Ref<IDxcCompiler3> compiler;
    if (!create || FAILED(create(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler.p)))) {
        std::printf("host: no dxcompiler.dll on PATH or in the Windows SDK, host comparison skipped\n");
        return false;
    }
    const auto ms = compile(compiler.p, L"ms", L"ms_6_5"), vs = compile(compiler.p, L"vs", L"vs_6_0"),
               ps = compile(compiler.p, L"ps", L"ps_6_0");
    if (ms.empty() || vs.empty() || ps.empty()) return false;
    Ref<IDXGIFactory6> factory;
    Ref<IDXGIAdapter1> adapter;
    Ref<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter.p))) ||
        FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device.p))))
        return false;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    adapter->GetDesc1(&adapter_desc);
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS9 options9{};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options7, sizeof(options7))) ||
        FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &options9, sizeof(options9))) ||
        options7.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED || !options9.MeshShaderPipelineStatsSupported) {
        std::printf("host %ls: no mesh-shader statistics, host comparison skipped\n", adapter_desc.Description);
        return false;
    }
    std::printf("host: %ls\n", adapter_desc.Description);
    Pipelines pipelines;
    CHECK(make_pipelines(device.p, {{ms.data(), ms.size()}, {vs.data(), vs.size()}, {ps.data(), ps.size()}},
                         pipelines));
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rt_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0},
                                D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    Ref<ID3D12Resource> target;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> rtv_heap;
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.p, nullptr, rtv);
    return run(device.p, pipelines, target.p, rtv, out);
}

void print(const char* who, const Results& r) {
    const char* const queries[] = {"VS draw", "DispatchMesh(2,2,1)", "DispatchMesh(1,1,2)", "never ended"};
    for (UINT q = 0; q < 4; ++q) {
        std::printf("%s STATISTICS1 %d (%s):", who, q, queries[q]);
        for (UINT i = 0; i < kWords; ++i) std::printf(" %s=%llu", kCounterNames[i], r.statistics1[q][i]);
        std::printf("\n");
    }
    std::printf("%s STATISTICS1 split (DispatchMesh 3 + 1):", who);
    for (UINT i = 0; i < kWords; ++i) std::printf(" %s=%llu", kCounterNames[i], r.split[i]);
    std::printf("\n%s STATISTICS (VS draw):", who);
    for (UINT i = 0; i < 11; ++i) std::printf(" %s=%llu", kCounterNames[i], r.statistics[i]);
    std::printf("\n");
}

// What must hold on any device: the classic counters of the two query types agree, the VS draw counts no AS/MS
// invocations, every mesh quad is two primitives, the split query sums to what one query over the same groups
// counts, the query never ended reads zero and nothing is written past the resolved range. Not checked here: the
// VS draw's MSPrimitives and the mesh draws' IA and VS counters, which NVIDIA's Vulkan driver counts and its D3D12
// driver does not (see main; docs/d3d12_vulkan.md section 4.6).
bool consistent(const Results& r) {
    for (UINT i = 0; i < 11; ++i) CHECK(r.statistics1[0][i] == r.statistics[i]);
    CHECK(r.statistics1[0][kAS] == 0 && r.statistics1[0][kMS] == 0);
    CHECK(r.statistics1[0][0] == 3 && r.statistics1[0][1] == 1);  // IAVertices, IAPrimitives
    CHECK(r.statistics1[1][kMSPrimitives] == 8 && r.statistics1[2][kMSPrimitives] == 4);
    CHECK(r.split[kMSPrimitives] == 8);
    CHECK(r.statistics1[1][kMS] > 0 && r.statistics1[2][kMS] * 2 == r.statistics1[1][kMS]);
    CHECK(r.split[kMS] == r.statistics1[1][kMS]);  // 4 groups, as query 1
    for (UINT q = 1; q < 3; ++q) CHECK(r.statistics1[q][kAS] == 0);
    for (UINT i = 0; i < kWords; ++i) CHECK(r.statistics1[3][i] == 0 && r.inner[i] == r.statistics1[1][i]);
    CHECK(r.past_end == 0xcdcdcdcdcdcdcdcdull);
    return true;
}

}  // namespace

int main() {
    vktest::Device d;
    d.request_mesh_shader = true;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    const auto ms_program = fssm_program({0x4d530001, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    const auto vs_program = fssm_program({0x56530002, 0xbf810000});
    const auto ms_container = shader_container(ms_program), ps_container = shader_container(ps_program);
    const auto vs_container = shader_container(vs_program);
    const auto library = std::filesystem::temp_directory_path() / "leonida_mesh_statistics_test";
    std::filesystem::create_directories(library);
    constexpr auto overwrite = std::filesystem::copy_options::overwrite_existing;
    std::filesystem::copy_file(LEONIDA_TEST_MS_SPV, library / "ms_draw.spv", overwrite);
    std::filesystem::copy_file(LEONIDA_TEST_PS_SPV, library / "ps_draw.spv", overwrite);
    std::filesystem::copy_file(LEONIDA_TEST_VS_FULLSCREEN_SPV, library / "vs_fullscreen.spv", overwrite);
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"ms_draw\": {\"stage\": \"ms\", \"fssm_sha256\": \"" << leonida::sha256_hex(ms_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [32, 1, 1], "
                 "\"max_vertices\": 4, \"max_primitives\": 2},\n"
                 " \"vs_fullscreen\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }
    const auto make_device = [&](bool queries, Ref<ID3D12Device>& device) {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.shader_dir = library.string();
        info.mesh_shader = d.mesh_shader;
        info.pipeline_statistics_query = d.pipeline_statistics_query;
        info.mesh_shader_queries = queries;
        ID3D12Device* inner = nullptr;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner)))) return false;
        if (FAILED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                     reinterpret_cast<void**>(&device.p))))
            return false;
        rt::d3d12::set_device_outer(inner, device.p);
        return true;
    };
    const auto stats_supported = [](ID3D12Device* device) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS9 options{};
        return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &options, sizeof(options))) &&
               options.MeshShaderPipelineStatsSupported;
    };

    // Without meshShaderQueries: not reported, and the heap is refused. The host refusal cannot be compared here:
    // WARP and the GPUs measured support these heaps (and refuse a count of 0 with E_INVALIDARG, as the layer does).
    {
        Ref<ID3D12Device> device;
        EXPECT(make_device(false, device));
        EXPECT(!stats_supported(device.p));
        Ref<ID3D12QueryHeap> heap;
        D3D12_QUERY_HEAP_DESC desc{D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1, 1, 0};
        EXPECT(device->CreateQueryHeap(&desc, IID_PPV_ARGS(&heap.p)) == E_NOTIMPL && !heap.p);
    }
    if (!d.mesh_shader_queries) {
        std::filesystem::remove_all(library);
        std::printf("no meshShaderQueries on this device: only the refusal without it was checked\n");
        return 77;
    }

    Results layer;
    {
        Ref<ID3D12Device> device;
        EXPECT(make_device(true, device));
        EXPECT(stats_supported(device.p));
        Ref<ID3D12QueryHeap> heap;
        D3D12_QUERY_HEAP_DESC desc{D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1, 0, 0};
        EXPECT(device->CreateQueryHeap(&desc, IID_PPV_ARGS(&heap.p)) == E_INVALIDARG && !heap.p);  // as the host
        const D3D12_SHADER_BYTECODE ms{ms_container.data(), ms_container.size()};
        const D3D12_SHADER_BYTECODE ps{ps_container.data(), ps_container.size()};
        const D3D12_SHADER_BYTECODE vs{vs_container.data(), vs_container.size()};
        Pipelines pipelines;
        EXPECT(make_pipelines(device.p, {ms, vs, ps}, pipelines));
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rt_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                    {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
        Ref<ID3D12Resource> target;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
            SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
        device->CreateRenderTargetView(target.p, nullptr, rtv);
        EXPECT(run(device.p, pipelines, target.p, rtv, layer));
    }
    std::filesystem::remove_all(library);
    print("layer", layer);
    Results host;
    const bool host_ran = run_host(host);
    if (host_ran) print("host ", host);
    EXPECT(consistent(layer));

    unsigned differ = 0;
    if (host_ran) {
        EXPECT(consistent(host));
        // Required: AS, MS and MSPrimitives of every mesh query, and AS, MS and the 11 classic counters of the VS
        // draw. Every other difference is printed: on NVIDIA (driver 610.88) the Vulkan driver counts the VS draw's
        // triangle in MESH_PRIMITIVES_GENERATED, and mesh groups and threads in IAVertices, IAPrimitives and
        // VSInvocations, which its D3D12 driver does not (an open finding of D3D-132).
        const auto compare = [&](const char* query, const UINT64* ours, const UINT64* theirs, bool mesh) {
            for (UINT i = 0; i < kWords; ++i) {
                if (ours[i] == theirs[i]) continue;
                const bool required = i == kAS || i == kMS || (mesh ? i == kMSPrimitives : i < 11);
                differ += required;
                std::printf("%s %s: layer %llu, host %llu%s\n", query, kCounterNames[i], ours[i], theirs[i],
                            required ? " (required to match)" : " (known Vulkan driver difference)");
            }
        };
        compare("VS draw", layer.statistics1[0].data(), host.statistics1[0].data(), false);
        compare("DispatchMesh(2,2,1)", layer.statistics1[1].data(), host.statistics1[1].data(), true);
        compare("DispatchMesh(1,1,2)", layer.statistics1[2].data(), host.statistics1[2].data(), true);
        compare("split", layer.split.data(), host.split.data(), true);
        if (differ) std::fprintf(stderr, "%u required counters differ from the host\n", differ);
    }
    EXPECT(!differ);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("D3D-132: PIPELINE_STATISTICS1 heaps refused without meshShaderQueries and created with it; VS and "
                "mesh draws counted, split and resolved at an offset%s\n",
                host.statistics1[1][kMSPrimitives] ? "; AS, MS and MSPrimitives match the host" : "");
    return 0;
}
