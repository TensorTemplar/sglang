"""
Unit tests for HTTP server admin auth.

Usage:
    python3 -m pytest test/test_http_server_auth.py -v
"""

import importlib.util
import os
import sys
import unittest


def _load_auth_module():
    """Load auth.py directly, avoiding importing the full sglang package.

    This keeps the test importable even if optional runtime deps (e.g. orjson/httpx)
    are not installed in the unit test environment.
    """
    this_dir = os.path.dirname(__file__)
    python_dir = os.path.abspath(os.path.join(this_dir, "..", ".."))
    auth_path = os.path.join(python_dir, "sglang", "srt", "utils", "auth.py")

    module_name = "_sglang_srt_utils_auth_for_test"
    spec = importlib.util.spec_from_file_location(module_name, auth_path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    # dataclasses (py3.12) may consult sys.modules during class processing
    sys.modules[module_name] = m
    spec.loader.exec_module(m)
    return m


_auth = _load_auth_module()
decide_request_auth = _auth.decide_request_auth
AuthLevel = _auth.AuthLevel


class TestHttpServerAdminAuth(unittest.TestCase):
    def _decide(
        self,
        *,
        method: str,
        path: str,
        authorization_header: str | None,
        x_api_key_header: str | None = None,
        api_key: str | None,
        admin_api_key: str | None,
        auth_level: AuthLevel,
    ):
        return decide_request_auth(
            method=method,
            path=path,
            authorization_header=authorization_header,
            x_api_key_header=x_api_key_header,
            api_key=api_key,
            admin_api_key=admin_api_key,
            auth_level=auth_level,
        )

    def test_no_keys_configured(self):
        # No keys configured -> NORMAL + ADMIN_OPTIONAL are open (legacy),
        # but ADMIN_FORCE must be rejected (403) explicitly.
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header=None,
                api_key=None,
                admin_api_key=None,
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header=None,
                api_key=None,
                admin_api_key=None,
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )

        d = self._decide(
            method="POST",
            path="/admin_force_demo",
            authorization_header=None,
            api_key=None,
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertFalse(d.allowed)
        self.assertEqual(d.error_status_code, 403)

    def test_api_key_only(self):
        # api_key configured -> NORMAL requires api_key (legacy).
        self.assertFalse(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header=None,
                api_key="user",
                admin_api_key=None,
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key=None,
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )

        # ADMIN_OPTIONAL requires api_key when only api_key is configured.
        self.assertFalse(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer wrong",
                api_key="user",
                admin_api_key=None,
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key=None,
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )

        # ADMIN_FORCE must be rejected even if api_key is configured (403).
        d = self._decide(
            method="POST",
            path="/admin_force_demo",
            authorization_header="Bearer user",
            api_key="user",
            admin_api_key=None,
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertFalse(d.allowed)
        self.assertEqual(d.error_status_code, 403)

    def test_admin_api_key_only(self):
        # admin_api_key only:
        # - normal endpoints open
        # - optional/force endpoints require admin_api_key
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer user",
                api_key=None,
                admin_api_key="admin",
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header=None,
                api_key=None,
                admin_api_key="admin",
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )

        # Optional endpoints require admin_api_key when admin_api_key is configured.
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer admin",
                api_key=None,
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )
        self.assertFalse(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer user",
                api_key=None,
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )

        d = self._decide(
            method="POST",
            path="/admin_force_demo",
            authorization_header="Bearer admin",
            api_key=None,
            admin_api_key="admin",
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertTrue(d.allowed)

    def test_with_both_api_keys(self):
        # both api_key and admin_api_key configured:
        # - normal endpoints require api_key
        # - optional endpoints require admin_api_key (api_key is NOT accepted)
        # - force endpoints require admin_api_key
        self.assertTrue(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        self.assertFalse(
            self._decide(
                method="GET",
                path="/v1/models",
                authorization_header="Bearer admin",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.NORMAL,
            ).allowed
        )
        # Optional endpoints must require admin_api_key when both keys are configured.
        self.assertFalse(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_optional_demo",
                authorization_header="Bearer admin",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_OPTIONAL,
            ).allowed
        )
        self.assertFalse(
            self._decide(
                method="POST",
                path="/admin_force_demo",
                authorization_header="Bearer user",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_FORCE,
            ).allowed
        )
        self.assertTrue(
            self._decide(
                method="POST",
                path="/admin_force_demo",
                authorization_header="Bearer admin",
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_FORCE,
            ).allowed
        )

    def test_options_is_always_allowed(self):
        # CORS preflight should never be blocked.
        self.assertTrue(
            self._decide(
                method="OPTIONS",
                path="/v1/models",
                authorization_header=None,
                api_key="user",
                admin_api_key="admin",
                auth_level=AuthLevel.ADMIN_FORCE,
            ).allowed
        )

    def test_health_and_metrics_are_always_allowed(self):
        # Health/metrics endpoints are always public by design, regardless of auth level / keys.
        combos = [
            dict(api_key=None, admin_api_key=None),
            dict(api_key="user", admin_api_key=None),
            dict(api_key=None, admin_api_key="admin"),
            dict(api_key="user", admin_api_key="admin"),
        ]
        paths_allowed = [
            "/health",
            "/health_generate",
            "/metrics",
            "/metrics/",
            "/metrics/prometheus",
        ]
        for keys in combos:
            for path in paths_allowed:
                self.assertTrue(
                    self._decide(
                        method="GET",
                        path=path,
                        authorization_header=None,
                        api_key=keys["api_key"],
                        admin_api_key=keys["admin_api_key"],
                        auth_level=AuthLevel.ADMIN_FORCE,
                    ).allowed,
                    msg=f"expected allowed for {path=} with {keys=}",
                )


class TestXApiKeyAuth(unittest.TestCase):
    """Tests for Anthropic-style x-api-key header authentication."""

    def _decide(self, **kwargs):
        return decide_request_auth(**kwargs)

    def test_x_api_key_accepted_for_normal_endpoint(self):
        d = self._decide(
            method="POST",
            path="/v1/messages",
            authorization_header=None,
            x_api_key_header="user",
            api_key="user",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(d.allowed)

    def test_x_api_key_rejected_when_wrong(self):
        d = self._decide(
            method="POST",
            path="/v1/messages",
            authorization_header=None,
            x_api_key_header="wrong",
            api_key="user",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertFalse(d.allowed)

    def test_bearer_takes_precedence_over_x_api_key(self):
        # Correct Bearer present -> allowed, regardless of x-api-key
        d = self._decide(
            method="POST",
            path="/v1/messages",
            authorization_header="Bearer user",
            x_api_key_header="wrong",
            api_key="user",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(d.allowed)

    def test_wrong_bearer_not_rescued_by_x_api_key(self):
        # Wrong Bearer present -> rejected even if x-api-key is correct
        d = self._decide(
            method="POST",
            path="/v1/messages",
            authorization_header="Bearer wrong",
            x_api_key_header="user",
            api_key="user",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertFalse(d.allowed)

    def test_x_api_key_fallback_when_authorization_absent(self):
        d = self._decide(
            method="POST",
            path="/v1/messages",
            authorization_header=None,
            x_api_key_header="user",
            api_key="user",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(d.allowed)

    def test_x_api_key_for_admin_optional(self):
        d = self._decide(
            method="POST",
            path="/admin_optional_demo",
            authorization_header=None,
            x_api_key_header="admin",
            api_key=None,
            admin_api_key="admin",
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertTrue(d.allowed)

    def test_x_api_key_for_admin_force(self):
        d = self._decide(
            method="POST",
            path="/admin_force_demo",
            authorization_header=None,
            x_api_key_header="admin",
            api_key=None,
            admin_api_key="admin",
            auth_level=AuthLevel.ADMIN_FORCE,
        )
        self.assertTrue(d.allowed)

    def test_no_credentials_rejected(self):
        d = self._decide(
            method="POST",
            path="/v1/messages",
            authorization_header=None,
            x_api_key_header=None,
            api_key="user",
            admin_api_key=None,
            auth_level=AuthLevel.NORMAL,
        )
        self.assertFalse(d.allowed)

    def test_x_api_key_with_both_keys_normal(self):
        # NORMAL endpoint requires api_key, x-api-key should work
        d = self._decide(
            method="POST",
            path="/v1/messages",
            authorization_header=None,
            x_api_key_header="user",
            api_key="user",
            admin_api_key="admin",
            auth_level=AuthLevel.NORMAL,
        )
        self.assertTrue(d.allowed)

    def test_x_api_key_with_both_keys_admin_optional(self):
        # ADMIN_OPTIONAL requires admin_api_key when both configured
        d = self._decide(
            method="POST",
            path="/admin_optional_demo",
            authorization_header=None,
            x_api_key_header="admin",
            api_key="user",
            admin_api_key="admin",
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertTrue(d.allowed)

        # User key should NOT work on admin_optional when admin_api_key configured
        d2 = self._decide(
            method="POST",
            path="/admin_optional_demo",
            authorization_header=None,
            x_api_key_header="user",
            api_key="user",
            admin_api_key="admin",
            auth_level=AuthLevel.ADMIN_OPTIONAL,
        )
        self.assertFalse(d2.allowed)


if __name__ == "__main__":
    unittest.main()
