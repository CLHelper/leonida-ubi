// ResolveQueryData is unconditional in D3D12, including un-ended slots zeroed via vkCmdFillBuffer. On Intel's
// Windows Vulkan driver, vkCmdFillBuffer after PredicateReducer::capture's compute dispatch is suppressed inside
// an active conditional rendering scope (D3D-148). Check that emit() suspends conditional rendering for the
// entire ResolveQueryData loop (not only before helper dispatches) and restores it afterward. Cases 5 and 6 check
// how resolves inside a pipeline-statistics query are replayed; cases 7 and 8 cover the other helper dispatches
// and their no-pool fallback (D3D-162). Cases 9 to 12 cover PIPELINE_STATISTICS1 (D3D-132): the primitives query
// beside the statistics one, the 14-UINT64 resolve layout, a split query summed over 13-counter segments, and an
// 11-counter resolve of such segments. No GPU or game data.
#include "../src/vulkan_replay.h"
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace leonida;

struct Trace {
    bool cond_active = false;
    unsigned cond_begins = 0;
    unsigned cond_ends = 0;
    unsigned fills = 0;
    unsigned fills_while_cond = 0;
    unsigned copies = 0;
    unsigned copies_while_cond = 0;
    unsigned binary_resolves = 0;
    unsigned binary_while_cond = 0;
    std::vector<std::string> events;
};

Trace* g_trace = nullptr;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

VKAPI_ATTR void VKAPI_CALL mock_begin_cond(VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {
    g_trace->cond_active = true;
    ++g_trace->cond_begins;
    g_trace->events.push_back("begin_cond");
}

VKAPI_ATTR void VKAPI_CALL mock_end_cond(VkCommandBuffer) {
    g_trace->cond_active = false;
    ++g_trace->cond_ends;
    g_trace->events.push_back("end_cond");
}

VKAPI_ATTR void VKAPI_CALL mock_fill_buffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, std::uint32_t) {
    ++g_trace->fills;
    if (g_trace->cond_active) ++g_trace->fills_while_cond;
    g_trace->events.push_back("fill");
}

std::string query_event(const char* name, VkQueryPool pool, std::uint32_t index) {
    return std::string(name) + " " + std::to_string(reinterpret_cast<std::uintptr_t>(pool)) + ":" +
           std::to_string(index);
}

// "copy_query pool:first+count @offset/stride"
VKAPI_ATTR void VKAPI_CALL mock_copy_query(VkCommandBuffer, VkQueryPool pool, std::uint32_t first,
                                           std::uint32_t count, VkBuffer, VkDeviceSize offset, VkDeviceSize stride,
                                           VkQueryResultFlags) {
    ++g_trace->copies;
    if (g_trace->cond_active) ++g_trace->copies_while_cond;
    g_trace->events.push_back(query_event("copy_query", pool, first) + "+" + std::to_string(count) + " @" +
                              std::to_string(offset) + "/" + std::to_string(stride));
}

VKAPI_ATTR void VKAPI_CALL mock_reset_query(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mock_begin_query(VkCommandBuffer, VkQueryPool pool, std::uint32_t index,
                                            VkQueryControlFlags) {
    g_trace->events.push_back(query_event("begin_query", pool, index));
}
VKAPI_ATTR void VKAPI_CALL mock_end_query(VkCommandBuffer, VkQueryPool pool, std::uint32_t index) {
    g_trace->events.push_back(query_event("end_query", pool, index));
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_pool(VkDevice, VkQueryPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL mock_write_timestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, std::uint32_t) {}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    ResourceBinding dst{};
    dst.buffer = handle<VkBuffer>(1);
    dst.buffer_offset = 24;
    dst.size = 128;

    ResourceBinding pred{};
    pred.buffer = handle<VkBuffer>(2);
    pred.size = 16;

    QueryPool heap{handle<VkQueryPool>(3), d3d12_query_heap::occlusion, 4, false, {false, true, false, true}, {}};
    QueryPool statistics_heap{handle<VkQueryPool>(7), d3d12_query_heap::pipeline_statistics, 2, false,
                              {false, false}, {}};
    // PIPELINE_STATISTICS1: statistics pool 13, primitives pool 14 (heap ID 12). Resolved into buffer 3 (512 bytes).
    QueryPool statistics1_heap{handle<VkQueryPool>(13), d3d12_query_heap::pipeline_statistics1, 2, false,
                               {false, false}, {}, handle<VkQueryPool>(14)};
    ResourceBinding big{};
    big.buffer = handle<VkBuffer>(15);
    big.size = 512;

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        if (id == 1) return &dst;
        if (id == 2) return &pred;
        if (id == 3) return &big;
        return nullptr;
    };

    ReplayFunctions vk{};
    vk.begin_conditional_rendering = mock_begin_cond;
    vk.end_conditional_rendering = mock_end_cond;
    vk.prepare_predicate = [](VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
        out = {source, offset, handle<VkBuffer>(4), 0, handle<VkDescriptorSet>(5)};
        return true;
    };
    vk.capture_predicate = [](VkCommandBuffer, const PredicateCapture&) { g_trace->events.push_back("capture"); };
    vk.reset_query_pool = mock_reset_query;
    vk.begin_query = mock_begin_query;
    vk.end_query = mock_end_query;
    vk.write_timestamp = mock_write_timestamp;
    vk.copy_query_pool_results = mock_copy_query;
    vk.fill_buffer = mock_fill_buffer;
    vk.query_pool = [&](QueryHeapId id) -> QueryPool* {
        return id == 1 ? &heap : id == 6 ? &statistics_heap : id == 12 ? &statistics1_heap : nullptr;
    };
    // "sum_statistics pool:first+count", then " segment->written @offset" unless both counter counts are 11.
    vk.sum_statistics = [](VkCommandBuffer, VkQueryPool pool, std::uint32_t first, std::uint32_t count,
                           std::uint32_t segment_counters, std::uint32_t counters, VkBuffer, VkDeviceSize offset) {
        std::string event = query_event("sum_statistics", pool, first) + "+" + std::to_string(count);
        if (segment_counters != 11 || counters != 11)
            event += " " + std::to_string(segment_counters) + "->" + std::to_string(counters) + " @" +
                     std::to_string(offset);
        g_trace->events.push_back(event);
    };
    vk.resolve_binary_occlusion = [](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer,
                                     VkDeviceSize) {
        ++g_trace->binary_resolves;
        if (g_trace->cond_active) ++g_trace->binary_while_cond;
        g_trace->events.push_back("binary");
    };

    // 1. Binary occlusion resolve with leading and trailing un-ended queries under a predicate:
    //    conditional rendering must end before the leading fill_buffer and resume after the trailing fill_buffer.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.set_predication({2, 0, d3d12_predication::equal_zero}), "record binary predication");
        expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 0, 3, 1, 16}), "record binary resolve");
        expect(list.close(), "close binary list");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "replay binary resolve");
        expect(trace.fills == 2 && trace.fills_while_cond == 0,
               "binary resolve: leading and trailing fill_buffer run outside conditional rendering");
        expect(trace.binary_resolves == 1 && trace.binary_while_cond == 0,
               "binary resolve: normalization dispatch runs outside conditional rendering");
        expect(trace.cond_begins == 2 && trace.cond_ends == 2 && !trace.cond_active,
               "binary resolve: conditional rendering suspended once and restored");
        const std::vector<std::string> want{"capture", "begin_cond", "end_cond", "fill",
                                            "binary",  "fill",       "begin_cond", "end_cond"};
        expect(trace.events == want, "binary resolve: exact suspend/resume event order");
    }

    // 2. Ordinary occlusion resolve with a leading un-ended query under a predicate (no helper dispatch):
    //    fill_buffer and copy_query_pool_results must also run outside conditional rendering.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.set_predication({2, 0, d3d12_predication::equal_zero}), "record ordinary predication");
        expect(list.resolve_queries({1, d3d12_query::occlusion, 0, 2, 1, 0}), "record ordinary resolve");
        expect(list.close(), "close ordinary list");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "replay ordinary resolve");
        expect(trace.fills == 1 && trace.fills_while_cond == 0,
               "ordinary resolve: fill_buffer runs outside conditional rendering");
        expect(trace.copies == 1 && trace.copies_while_cond == 0,
               "ordinary resolve: copy_query_pool_results runs outside conditional rendering");
        expect(trace.cond_begins == 2 && trace.cond_ends == 2 && !trace.cond_active,
               "ordinary resolve: conditional rendering suspended once and restored");
    }

    // 3. Zero-count resolve under a predicate does not toggle conditional rendering unnecessarily.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.set_predication({2, 0, d3d12_predication::equal_zero}), "record empty resolve predication");
        expect(list.resolve_queries({1, d3d12_query::occlusion, 0, 0, 1, 0}), "record empty resolve");
        expect(list.close(), "close empty resolve list");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "replay empty resolve");
        expect(trace.fills == 0 && trace.copies == 0 && trace.cond_begins == 1 && trace.cond_ends == 1,
               "empty resolve: no redundant conditional rendering toggle");
    }

    // 4. Unpredicated resolve never touches conditional rendering.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 0, 3, 1, 16}), "record unpredicated resolve");
        expect(list.close(), "close unpredicated list");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "replay unpredicated resolve");
        expect(trace.fills == 2 && trace.binary_resolves == 1 && trace.cond_begins == 0 && trace.cond_ends == 0,
               "unpredicated resolve: conditional rendering untouched");
    }

    // 5. D3D-162 without a statistics segment pool (the single-list replay()): a binary resolve that dispatches
    //    inside a pipeline-statistics query is still refused, while one that only writes zeros is replayed.
    {
        Trace trace;
        g_trace = &trace;
        CommandList dispatches, zeros;
        for (auto* list : {&dispatches, &zeros}) {
            const std::uint32_t index = list == &dispatches ? 1 : 0;  // query 1 was ended, query 0 was not
            expect(list->begin_query({6, d3d12_query::pipeline_statistics, 0}), "record statistics begin");
            expect(list->resolve_queries({1, d3d12_query::binary_occlusion, index, 1, 1, 0}), "record inner resolve");
            expect(list->end_query({6, d3d12_query::pipeline_statistics, 0}), "record statistics end");
            expect(list->close(), "close statistics list");
        }
        expect(replay(dispatches, VK_NULL_HANDLE, resolve, vk) == ReplayResult::unsupported,
               "no segment pool: a binary dispatch inside pipeline statistics is refused");
        expect(replay(zeros, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success,
               "no segment pool: a zero-fill binary resolve inside pipeline statistics is replayed");
        expect(trace.binary_resolves == 0 && trace.fills == 1, "no segment pool: only the zero fill was recorded");
    }

    // 6. D3D-162 with a statistics segment pool: the statistics query is split around the resolve inside it, so
    //    the normalisation dispatch runs between two segments, and resolving the query sums the segments.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "record split begin");
        expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 1, 1, 1, 0}), "record split inner resolve");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "record split end");
        expect(list.resolve_queries({6, d3d12_query::pipeline_statistics, 0, 1, 1, 0}), "record split resolve");
        expect(list.close(), "close split list");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare split list");
        expect(prepared.statistics_segment_count == 2, "split list: two statistics segments");
        auto segments = std::make_shared<OcclusionSegmentPool>();
        segments->pool = handle<VkQueryPool>(9);
        segments->destroy = mock_destroy_pool;
        QueryReplay queries;
        expect(queries.plan(prepared, {}, 0, true, segments, 4) == ReplayResult::success, "plan split list");
        expect(queries.needs_binary && queries.needs_statistics_sum, "split list: binary and statistics helpers");
        emit(prepared, VK_NULL_HANDLE, vk);
        queries.commit();
        const std::vector<std::string> want{"begin_query 9:4", "end_query 9:4",  "binary",
                                            "begin_query 9:5", "end_query 9:5", "sum_statistics 9:4+2"};
        expect(trace.events == want, "split list: segments end before and begin after the helper dispatch");
        expect(statistics_heap.occlusion.count(0) && statistics_heap.occlusion[0].count == 2,
               "split list: the heap keeps the query's segments for later resolves");
    }

    // 7. All non-query helpers and a resolve share the segment numbering. Exercise both footprint
    // directions/formats and multiple conversion dispatches under an active predicate.
    for (bool rgb96 : {false, true}) for (bool upload : {false, true}) {
        Trace trace;
        g_trace = &trace;
        auto helpers = vk;
        unsigned allocations = 0;
        helpers.prepare_predicate = [&](VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
            ++allocations;
            out = {source, offset, handle<VkBuffer>(4), 0, handle<VkDescriptorSet>(5)};
            return true;
        };
        helpers.prepare_predicated_copy = [&](VkBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, VkDeviceSize,
                                               std::vector<PredicatedCopyDispatch>& jobs) {
            ++allocations;
            jobs.push_back({});
            return true;
        };
        helpers.emit_predicated_copy = [](VkCommandBuffer, const std::vector<PredicatedCopyDispatch>&) {
            expect(g_trace->cond_active, "copy helper remains predicated");
            g_trace->events.push_back("predicated_copy");
        };
        helpers.depth24_prepare = [&](Depth24Job& job, std::vector<Depth24Job>& parts) {
            ++allocations;
            job.scratch = handle<VkBuffer>(10);
            parts.assign(2, job);
            return true;
        };
        helpers.depth24_run = [](VkCommandBuffer, const Depth24Job&) {
            expect(!g_trace->cond_active, "footprint helper is unconditional");
            g_trace->events.push_back("footprint");
        };
        helpers.pipeline_barrier = +[](VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                       VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                       const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {};
        helpers.copy_buffer = +[](VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {};
        helpers.copy_image = +[](VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                 std::uint32_t, const VkImageCopy*) {};
        helpers.copy_buffer_to_image = +[](VkCommandBuffer, VkBuffer, VkImage, VkImageLayout,
                                           std::uint32_t, const VkBufferImageCopy*) {
            g_trace->events.push_back("upload");
        };
        helpers.copy_image_to_buffer = +[](VkCommandBuffer, VkImage, VkImageLayout, VkBuffer,
                                           std::uint32_t, const VkBufferImageCopy*) {
            g_trace->events.push_back("readback");
        };
        ResourceBinding buffer = dst;
        buffer.buffer_offset = 0;
        buffer.size = 512;
        ResourceBinding image{};
        image.image = handle<VkImage>(11);
        image.format = rgb96 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_D32_SFLOAT_S8_UINT;
        image.aspects = rgb96 ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        image.width = image.height = image.depth = image.mip_levels = image.array_layers = image.samples = 1;
        image.rgb96 = rgb96;
        image.depth24 = image.depth24_as_float = !rgb96;
        const auto resources = [&](ResourceId id) -> const ResourceBinding* {
            return id == 1 ? &buffer : id == 2 ? &pred : id == 3 ? &image : nullptr;
        };
        const CopyLocation texture{3, false, 0, 0, 0, 0, 0, 0};
        const CopyLocation footprint{1, true, 0, 0, 1, 1, 1, 256};
        CommandList list;
        expect(list.begin_query({6, d3d12_query::pipeline_statistics, 0}), "mixed begin");
        expect(list.set_predication({2, 0, d3d12_predication::equal_zero}), "mixed capture");
        expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 1, 1, 1, 0}), "mixed resolve");
        expect(list.copy_buffer({1, 0, 2, 0, 8}), "mixed predicated copy");
        expect(list.copy_texture({upload ? texture : footprint, 0, 0, 0,
                                  upload ? footprint : texture, false, {}}), "mixed footprint copy");
        expect(list.end_query({6, d3d12_query::pipeline_statistics, 0}), "mixed end");
        expect(list.resolve_queries({6, d3d12_query::pipeline_statistics, 0, 1, 1, 128}), "mixed sum");
        expect(list.close(), "close mixed list");
        PreparedList prepared;
        expect(prepare(list, resources, helpers, prepared, true) == ReplayResult::success, "prepare mixed helpers");
        expect(prepared.statistics_segment_count == 5 && allocations == 3, "five shared segments");
        QueryReplay no_pool;
        expect(no_pool.plan(prepared, {}, 0, true) == ReplayResult::unsupported,
               "plan refuses helper markers without statistics pool");
        expect(trace.events.empty(), "no-pool plan emits nothing");
        auto pool = std::make_shared<OcclusionSegmentPool>();
        pool->pool = handle<VkQueryPool>(9);
        pool->destroy = mock_destroy_pool;
        QueryReplay queries;
        expect(queries.plan(prepared, {}, 0, true, pool, 4) == ReplayResult::success, "plan mixed helpers");
        expect(queries.needs_binary && queries.needs_statistics_sum, "mixed helpers require statistics sum");
        emit(prepared, VK_NULL_HANDLE, helpers);
        std::vector<std::string> want{
            "begin_query 9:4", "end_query 9:4", "capture", "begin_query 9:5", "begin_cond",
            "end_query 9:5", "end_cond", "binary", "begin_cond", "begin_query 9:6",
            "end_query 9:6", "predicated_copy", "begin_query 9:7"};
        if (!upload) want.push_back("readback");
        const std::vector<std::string> conversion{
            "end_query 9:7", "end_cond", "footprint", "begin_cond", "end_cond", "footprint", "begin_cond",
            "begin_query 9:8"};
        want.insert(want.end(), conversion.begin(), conversion.end());
        if (upload) want.push_back("upload");
        const std::vector<std::string> tail{
            "end_query 9:8", "end_cond", "sum_statistics 9:4+5", "begin_cond", "end_cond"};
        want.insert(want.end(), tail.begin(), tail.end());
        expect(trace.events == want, "mixed helpers: exact segment and conditional scope order");

        // 8. Fallback checks each helper separately, before its scratch allocation and before emission.
        for (unsigned kind = 0; kind != 3; ++kind) {
            CommandList fallback;
            if (kind == 1) expect(fallback.set_predication({2, 0, 0}), "fallback pre-query predicate");
            expect(fallback.begin_query({6, d3d12_query::pipeline_statistics, 0}), "fallback begin");
            if (kind == 0) expect(fallback.set_predication({2, 0, 0}), "fallback capture");
            if (kind == 1) expect(fallback.copy_buffer({1, 0, 2, 0, 8}), "fallback copy");
            if (kind == 2) expect(fallback.copy_texture({upload ? texture : footprint, 0, 0, 0,
                                                       upload ? footprint : texture, false, {}}), "fallback footprint");
            expect(fallback.end_query({6, d3d12_query::pipeline_statistics, 0}) && fallback.close(), "fallback end");
            allocations = 0;
            trace.events.clear();
            expect(replay(fallback, VK_NULL_HANDLE, resources, helpers) == ReplayResult::unsupported,
                   "single-list fallback refuses helper");
            expect(trace.events.empty() && allocations == (kind == 1 ? 1u : 0u),
                   "fallback refuses before helper scratch and emits nothing");
            expect(prepare(fallback, resources, helpers, prepared, true) == ReplayResult::success,
                   "prepare fallback for explicit pool check");
            QueryReplay missing;
            expect(missing.plan(prepared) == ReplayResult::unsupported, "each helper needs a statistics pool");
        }
    }

    // 9. PIPELINE_STATISTICS1 without a split: the primitives query runs beside the statistics query, and the resolve
    //    writes 14 UINT64s per query, 112 bytes apart: the 13 statistics from pool 13, MSPrimitives from pool 14 at
    //    +104, and zeros for the query that was never ended.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.begin_query({12, d3d12_query::pipeline_statistics1, 0}), "record statistics1 begin");
        expect(list.end_query({12, d3d12_query::pipeline_statistics1, 0}), "record statistics1 end");
        expect(list.resolve_queries({12, d3d12_query::pipeline_statistics1, 0, 2, 3, 16}),
               "record statistics1 resolve");
        expect(list.close(), "close statistics1 list");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "replay statistics1");
        const std::vector<std::string> want{"begin_query 14:0", "begin_query 13:0", "end_query 14:0",
                                            "end_query 13:0", "copy_query 13:0+1 @16/112",
                                            "copy_query 14:0+1 @120/112", "fill"};
        expect(trace.events == want, "statistics1: both queries run, and the resolve lays out 14 UINT64s");
        expect(trace.fills == 1, "statistics1: the query never ended reads zero");
    }

    // 10. A split PIPELINE_STATISTICS1 query: the statistics run in 13-counter segments around the helper dispatch,
    //     the primitives query is not split (helpers make no primitives), and the resolve sums the segments into the
    //     first 13 UINT64s and copies MSPrimitives after them.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.begin_query({12, d3d12_query::pipeline_statistics1, 1}), "record split statistics1 begin");
        expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 1, 1, 1, 0}), "record split inner resolve");
        expect(list.end_query({12, d3d12_query::pipeline_statistics1, 1}), "record split statistics1 end");
        expect(list.resolve_queries({12, d3d12_query::pipeline_statistics1, 1, 1, 3, 0}), "record split resolve");
        expect(list.close(), "close split statistics1 list");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare split statistics1");
        expect(prepared.statistics_segment_count == 2 && prepared.statistics1_segments,
               "split statistics1: two segments that must count task and mesh invocations");
        auto segments = std::make_shared<OcclusionSegmentPool>();
        segments->pool = handle<VkQueryPool>(9);
        segments->destroy = mock_destroy_pool;
        segments->counters = 13;
        QueryReplay queries;
        expect(queries.plan(prepared, {}, 0, true, segments, 4) == ReplayResult::success, "plan split statistics1");
        emit(prepared, VK_NULL_HANDLE, vk);
        queries.commit();
        const std::vector<std::string> want{"begin_query 14:1", "begin_query 9:4", "end_query 9:4", "binary",
                                            "begin_query 9:5",  "end_query 14:1", "end_query 9:5",
                                            "sum_statistics 9:4+2 13->13 @0", "copy_query 14:1+1 @104/112"};
        expect(trace.events == want, "split statistics1: segments summed, primitives copied after them");
    }

    // 11. An 11-counter PIPELINE_STATISTICS query split into a 13-counter segment pool (one that also holds a
    //     PIPELINE_STATISTICS1 query's segments) writes only its 11 sums; 11-counter segments cannot resolve a
    //     PIPELINE_STATISTICS1 query.
    for (const bool mesh : {false, true}) {
        Trace trace;
        g_trace = &trace;
        const QueryHeapId id = mesh ? 12 : 6;
        const std::uint32_t type = mesh ? d3d12_query::pipeline_statistics1 : d3d12_query::pipeline_statistics;
        CommandList list;
        expect(list.begin_query({id, type, 0}), "record mixed-width begin");
        expect(list.resolve_queries({1, d3d12_query::binary_occlusion, 1, 1, 1, 0}), "record mixed-width inner");
        expect(list.end_query({id, type, 0}), "record mixed-width end");
        expect(list.resolve_queries({id, type, 0, 1, 3, 0}), "record mixed-width resolve");
        expect(list.close(), "close mixed-width list");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare mixed-width list");
        auto segments = std::make_shared<OcclusionSegmentPool>();
        segments->pool = handle<VkQueryPool>(9);
        segments->destroy = mock_destroy_pool;
        segments->counters = mesh ? 11 : 13;
        QueryReplay queries;
        const ReplayResult planned = queries.plan(prepared, {}, 0, true, segments, 4);
        if (mesh) {
            expect(planned == ReplayResult::unsupported, "11-counter segments cannot resolve PIPELINE_STATISTICS1");
            continue;
        }
        expect(planned == ReplayResult::success, "plan 11-counter resolve of 13-counter segments");
        emit(prepared, VK_NULL_HANDLE, vk);
        expect(!trace.events.empty() && trace.events.back() == "sum_statistics 9:4+2 13->11 @0",
               "11-counter resolve of 13-counter segments writes 11 sums");
    }

    // 12. A PIPELINE_STATISTICS1 heap without its primitives pool cannot run the query.
    {
        QueryPool broken = statistics1_heap;
        broken.primitives = VK_NULL_HANDLE;
        auto missing = vk;
        missing.query_pool = [&](QueryHeapId id) -> QueryPool* { return id == 12 ? &broken : nullptr; };
        CommandList list;
        expect(list.begin_query({12, d3d12_query::pipeline_statistics1, 0}), "record primitives-less begin");
        expect(list.end_query({12, d3d12_query::pipeline_statistics1, 0}) && list.close(),
               "record primitives-less end");
        expect(replay(list, VK_NULL_HANDLE, resolve, missing) == ReplayResult::unsupported,
               "statistics1 without a primitives pool is unsupported");
    }

    if (failures) return 1;
    std::puts("replay_query_predication_test: query resolves suspend conditional rendering around fills, copies and dispatches");
    return 0;
}
