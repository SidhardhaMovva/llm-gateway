"""Real tests for per-tenant fixed-window rate limiting (fake Redis)."""
from app.rate_limiter import check_rate_limit, get_remaining


def test_requests_allowed_up_to_the_limit(fake_redis, run):
    results = [run(check_rate_limit("tenant-a", 3)) for _ in range(3)]
    assert results == [(True, 1), (True, 2), (True, 3)]


def test_request_over_the_limit_is_rejected(fake_redis, run):
    for _ in range(3):
        run(check_rate_limit("tenant-a", 3))
    allowed, count = run(check_rate_limit("tenant-a", 3))
    assert allowed is False
    assert count == 4


def test_limits_are_per_tenant(fake_redis, run):
    run(check_rate_limit("tenant-a", 1))
    allowed_a, _ = run(check_rate_limit("tenant-a", 1))
    allowed_b, count_b = run(check_rate_limit("tenant-b", 1))
    assert allowed_a is False
    assert (allowed_b, count_b) == (True, 1)


def test_non_positive_limit_means_unlimited(fake_redis, run):
    assert run(check_rate_limit("tenant-a", 0)) == (True, 0)
    assert run(check_rate_limit("tenant-a", -5)) == (True, 0)
    # The unlimited path must not touch Redis at all.
    assert fake_redis.data == {}


def test_window_expiry_is_set_once_on_first_request(fake_redis, run):
    run(check_rate_limit("tenant-a", 10))
    run(check_rate_limit("tenant-a", 10))
    rate_calls = [c for c in fake_redis.expire_calls if c[0].endswith("tenant-a")]
    assert rate_calls == [("llm_gateway:rate:tenant-a", 60)]


def test_get_remaining_counts_down(fake_redis, run):
    assert run(get_remaining("tenant-a", 5)) == 5
    run(check_rate_limit("tenant-a", 5))
    run(check_rate_limit("tenant-a", 5))
    assert run(get_remaining("tenant-a", 5)) == 3


def test_get_remaining_never_goes_negative(fake_redis, run):
    fake_redis.data["llm_gateway:rate:tenant-a"] = "99"
    assert run(get_remaining("tenant-a", 5)) == 0
