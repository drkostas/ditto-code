from understudy import ladder
from understudy.memory import GB, parse_meminfo, parse_vm_stat

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               65536.
Pages active:                            400000.
Pages inactive:                          131072.
Pages speculative:                         8192.
Pages throttled:                              0.
Pages wired down:                        480000.
Pages purgeable:                           8192.
"""


def test_vm_stat_counts_only_what_can_be_handed_out_without_swapping():
    # free + inactive + speculative + purgeable = 212992 pages of 16 KiB = 3.25 GiB
    assert parse_vm_stat(VM_STAT) == 212992 * 16384
    assert parse_vm_stat("not vm_stat") is None
    assert parse_vm_stat(VM_STAT.replace("Pages purgeable", "Pages other")) is None


def test_meminfo():
    assert parse_meminfo("MemTotal: 100 kB\nMemAvailable:   2048 kB\n") == 2048 * 1024
    assert parse_meminfo("MemTotal: 100 kB\n") is None


def tags(*rows):
    return [{"name": n, "digest": d, "size": int(s * GB)} for n, d, s in rows]


def test_ladder_is_the_prefixed_models_largest_first_and_estimates_say_so(tmp_path):
    m = ladder.Measurements(tmp_path / "m.json")
    t = tags(("understudy/small", "a", 6), ("other/big", "b", 40), ("understudy/big", "c", 20))
    r = ladder.rungs(t, "understudy/", m)
    assert [x.name for x in r] == ["understudy/big", "understudy/small"]
    assert not r[0].measured and r[0].need > r[0].disk


def test_a_measurement_replaces_the_estimate_only_for_the_same_digest(tmp_path):
    m = ladder.Measurements(tmp_path / "m.json")
    m.put("understudy/small", "a", int(7 * GB))
    assert ladder.rungs(tags(("understudy/small", "a", 6)), "understudy/", m)[0].need == int(7 * GB)
    # a re-pulled model has a new digest, so the old measurement no longer applies
    assert not ladder.rungs(tags(("understudy/small", "z", 6)), "understudy/", m)[0].measured
    # the largest size seen is kept: the context grows during a chat
    assert m.put("understudy/small", "a", int(6 * GB)) is False


def rung(name, need, context=262144, disk=None):
    return ladder.Rung(name, "d", disk if disk is not None else need, need, True, context)


LADDER = [rung("big", int(23 * GB)), rung("mid", int(18 * GB)), rung("small", int(8 * GB))]


def test_the_largest_rung_that_fits_with_headroom():
    c = ladder.choose(LADDER, int(21 * GB), {}, int(2 * GB), int(25 * GB))
    assert c.rung.name == "mid" and c.reason == "largest that fits"
    c = ladder.choose(LADDER, int(30 * GB), {}, int(2 * GB), int(25 * GB))
    assert c.rung.name == "big"


def test_the_gpu_budget_rules_out_a_rung_that_would_fit_in_memory():
    c = ladder.choose(LADDER, int(30 * GB), {}, int(2 * GB), int(20 * GB))
    assert c.rung.name == "mid"


def test_a_loaded_rung_keeps_the_request_even_when_a_bigger_one_now_fits():
    c = ladder.choose(LADDER, int(40 * GB), {"small": int(8 * GB)}, int(2 * GB), None)
    assert c.rung.name == "small" and c.reason == "already loaded"


def test_memory_held_by_ollama_counts_as_available_for_the_next_choice():
    # 4 GB free, plus 16 GB held by a model that is not on the ladder: Ollama releases it first
    c = ladder.choose(LADDER, int(4 * GB), {"someone-else": int(16 * GB)}, int(2 * GB), None)
    assert c.rung.name == "mid"


def test_nothing_fits_says_why_and_never_loads_blind():
    c = ladder.choose(LADDER, int(5 * GB), {}, int(2 * GB), None)
    assert c.rung is None and "not enough free memory" in c.reason and "small" in c.reason
    c = ladder.choose(LADDER, None, {}, int(2 * GB), None)
    assert c.rung is None and "could not be read" in c.reason
    assert ladder.choose([], int(64 * GB), {}, 0, None).rung is None


def test_the_window_comes_from_ollama_or_else_the_tag():
    assert ladder.num_ctx("temperature 0.6\nnum_ctx 65536\n") == 65536
    assert ladder.num_ctx("temperature 0.6") is None
    assert ladder.context_from_tag("understudy/qwen3.5-9b:128k") == 131072
    assert ladder.context_from_tag("understudy/qwen3.5-9b") is None


# one model in three windows: the same weights on disk, more context cache for a bigger window
WINDOWS = [rung("q9:64k", int(8 * GB), 65536, int(6.6 * GB)),
           rung("q9:128k", int(9 * GB), 131072, int(6.6 * GB)),
           rung("q9:256k", int(11.5 * GB), 262144, int(6.6 * GB)),
           rung("q35:64k", int(19 * GB), 65536, int(17 * GB)),
           rung("q35:256k", int(21 * GB), 262144, int(17 * GB))]


def test_the_largest_model_at_the_smallest_window_that_holds_the_conversation():
    c = ladder.choose(WINDOWS, int(30 * GB), {}, int(2 * GB), None, needed=30_000)
    assert c.rung.name == "q35:64k"
    c = ladder.choose(WINDOWS, int(30 * GB), {}, int(2 * GB), None, needed=140_000)
    assert c.rung.name == "q35:256k"
    # not enough for the big model: the small one, still at the smallest window that holds it
    c = ladder.choose(WINDOWS, int(12 * GB), {}, int(2 * GB), None, needed=100_000)
    assert c.rung.name == "q9:128k"


def test_a_loaded_rung_whose_window_is_too_small_gives_way():
    c = ladder.choose(WINDOWS, int(12 * GB), {"q9:64k": int(8 * GB)}, int(2 * GB), None,
                      needed=100_000)
    assert c.rung.name == "q9:128k" and c.reason == "largest that fits"
    c = ladder.choose(WINDOWS, int(1 * GB), {"q9:64k": int(8 * GB)}, int(2 * GB), None,
                      needed=30_000)
    assert c.rung.name == "q9:64k" and c.reason == "already loaded"


def test_a_conversation_longer_than_every_window_is_refused_with_the_numbers():
    c = ladder.choose(WINDOWS, int(64 * GB), {}, int(2 * GB), None, needed=300_000)
    assert c.rung is None and "300000" in c.reason and "262144" in c.reason


def qwen35(blocks, kv, **more):
    """/api/show model_info as Ollama gives it for Qwen3.5 (values read from each GGUF's header)."""
    return {"general.architecture": "qwen35", "qwen35.block_count": blocks,
            "qwen35.attention.head_count_kv": kv, "qwen35.attention.key_length": 256,
            "qwen35.attention.value_length": 256, "qwen35.full_attention_interval": 4, **more}


def test_the_cache_a_token_keeps_comes_from_the_models_attention_layout():
    # 2B: 24 layers, every fourth keeps a cache of 2 KV heads; 4B and 9B: 32 layers of 4 KV heads
    assert ladder.cache_elements(qwen35(24, 2)) == 6 * 2 * 512 == 6144
    assert ladder.cache_elements(qwen35(32, 4)) == 8 * 4 * 512 == 16384
    # the 9B at 1.25 bytes an element is the 20 KB the estimate always used for it
    assert int(16384 * ladder.CACHE_BYTES_PER_ELEMENT) == ladder.ESTIMATE_PER_TOKEN
    # the layers' own flags win over the interval; a KV head count per layer is summed
    flags = [True, True, True, False] * 6
    assert ladder.cache_elements(qwen35(24, 2, **{"qwen35.attention.recurrent_layers": flags})) == 6144
    assert ladder.cache_elements(qwen35(4, [0, 0, 0, 2], **{"qwen35.full_attention_interval": None})) == 1024
    # a dense model keeps a cache on every layer
    dense = {"general.architecture": "llama", "llama.block_count": 32, "llama.attention.head_count_kv": 8,
             "llama.attention.key_length": 128, "llama.attention.value_length": 128}
    assert ladder.cache_elements(dense) == 32 * 8 * 256
    # anything it cannot read is None, and the estimate keeps its 20 KB
    assert ladder.cache_elements(None) is None
    assert ladder.cache_elements({"general.architecture": "qwen35"}) is None
    assert ladder.cache_elements(qwen35(24, [])) is None
    assert ladder.cache_elements(qwen35(24, True)) is None


def test_a_small_model_is_priced_by_its_own_cache_and_takes_the_request_a_9b_cannot(tmp_path):
    # ⭐ 2026-10-24: 5.9 GB could be freed and the 9B was the smallest rung, so a local chat was
    # refused. A 2B (1.9 GB) at 64k was priced with the 9B's 20 KB a token and refused too.
    m = ladder.Measurements(tmp_path / "m.json")
    t = tags(("understudy/qwen3.5-9b:64k", "a", 6.6), ("understudy/qwen3.5-4b:64k", "b", 3.3),
             ("understudy/qwen3.5-2b:64k", "c", 1.95))
    per = {"understudy/qwen3.5-9b:64k": int(16384 * 1.25), "understudy/qwen3.5-4b:64k": int(16384 * 1.25),
           "understudy/qwen3.5-2b:64k": int(6144 * 1.25)}
    priced = ladder.rungs(t, "understudy/", m, per_token=per)
    c = ladder.choose(priced, int(5.9 * GB), {}, int(2 * GB), None, needed=40_000)
    assert c.rung.name == "understudy/qwen3.5-2b:64k" and c.reason == "largest that fits"
    # without the model's own layout every rung keeps the old 20 KB, and nothing fits
    assert ladder.choose(ladder.rungs(t, "understudy/", m), int(5.9 * GB), {}, int(2 * GB), None,
                         needed=40_000).rung is None
    # with more to spare the largest that fits still wins: the 4B is not underpriced by its size
    assert ladder.choose(priced, int(8 * GB), {}, int(2 * GB), None, needed=40_000).rung.name \
        == "understudy/qwen3.5-4b:64k"
    four = next(r for r in priced if r.name.endswith("4b:64k"))
    assert four.need >= int(3.3 * GB) + 65536 * 16384
