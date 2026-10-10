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


def rung(name, need):
    return ladder.Rung(name, "d", need, need, True)


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
