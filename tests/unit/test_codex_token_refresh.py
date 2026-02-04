"""Unit tests for Codex token refresh mechanism (8-day interval)."""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anthropic_proxy.codex import TOKEN_REFRESH_INTERVAL, CodexAuth


@pytest.fixture
def codex_auth(tmp_path: Path) -> CodexAuth:
    """Create a CodexAuth instance with temporary auth file."""
    auth_file = tmp_path / "auth.json"
    # Pre-create file to avoid issues
    auth_file.write_text("{}")

    with patch("anthropic_proxy.config_manager.DEFAULT_AUTH_FILE", auth_file):
        auth = CodexAuth()
        return auth


class TestIsTokenStale:
    """Tests for _is_token_stale method."""

    def test_returns_true_when_last_refresh_missing(self, codex_auth: CodexAuth) -> None:
        """Token is considered stale if last_refresh is not recorded."""
        codex_auth._auth_data = {"access": "test_token"}
        assert codex_auth._is_token_stale() is True

    def test_returns_true_when_token_older_than_7_days(self, codex_auth: CodexAuth) -> None:
        """Token is stale if last_refresh is more than 7 days ago."""
        eight_days_ago = int(time.time()) - (8 * 24 * 60 * 60)
        codex_auth._auth_data = {"access": "test_token", "last_refresh": eight_days_ago}
        assert codex_auth._is_token_stale() is True

    def test_returns_false_when_token_within_7_days(self, codex_auth: CodexAuth) -> None:
        """Token is not stale if last_refresh is within 7 days."""
        six_days_ago = int(time.time()) - (6 * 24 * 60 * 60)
        codex_auth._auth_data = {"access": "test_token", "last_refresh": six_days_ago}
        assert codex_auth._is_token_stale() is False

    def test_returns_false_when_token_just_refreshed(self, codex_auth: CodexAuth) -> None:
        """Token is not stale if just refreshed (now)."""
        now = int(time.time())
        codex_auth._auth_data = {"access": "test_token", "last_refresh": now}
        assert codex_auth._is_token_stale() is False

    def test_exactly_at_7_days_boundary(self, codex_auth: CodexAuth) -> None:
        """Token at exactly 7 days boundary should not be stale (> vs >=)."""
        # Use current time as reference to avoid floating point issues
        current_time = int(time.time())
        # At exact boundary: time.time() == last_refresh + INTERVAL
        # So time.time() > (last_refresh + INTERVAL) should be False
        last_refresh = current_time - TOKEN_REFRESH_INTERVAL
        codex_auth._auth_data = {"access": "test_token", "last_refresh": last_refresh}
        # We need to patch time.time to return the exact value we expect
        with patch("anthropic_proxy.codex.time.time", return_value=current_time):
            result = codex_auth._is_token_stale()
        assert result is False

    def test_one_second_over_7_days_is_stale(self, codex_auth: CodexAuth) -> None:
        """Token 1 second over 7 days is stale."""
        seven_days_plus_one = int(time.time()) - TOKEN_REFRESH_INTERVAL - 1
        codex_auth._auth_data = {"access": "test_token", "last_refresh": seven_days_plus_one}
        assert codex_auth._is_token_stale() is True


class TestGetAccessToken:
    """Tests for get_access_token method with stale checking."""

    @pytest.mark.asyncio
    async def test_refreshes_when_token_stale(self, codex_auth: CodexAuth) -> None:
        """Should refresh token when 7+ days since last_refresh."""
        eight_days_ago = int(time.time()) - (8 * 24 * 60 * 60)
        # Don't use _save(), just set _auth_data directly
        codex_auth._auth_data = {
            "access": "old_token",
            "refresh": "refresh_token",
            "expires": int(time.time()) + 3600,  # Not expired
            "last_refresh": eight_days_ago,  # But stale
        }

        with patch.object(codex_auth, "_refresh_token", new_callable=AsyncMock) as mock_refresh:
            mock_refresh.return_value = "new_token"
            # Patch _load to prevent overwriting _auth_data
            with patch.object(codex_auth, "_load"):
                result = await codex_auth.get_access_token()

        assert result == "new_token"
        mock_refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_refreshes_when_token_expired(self, codex_auth: CodexAuth) -> None:
        """Should refresh token when expired (even if not stale)."""
        one_day_ago = int(time.time()) - (24 * 60 * 60)
        codex_auth._auth_data = {
            "access": "old_token",
            "refresh": "refresh_token",
            "expires": int(time.time()) - 10,  # Expired 10 seconds ago
            "last_refresh": one_day_ago,  # Not stale
        }

        with patch.object(codex_auth, "_refresh_token", new_callable=AsyncMock) as mock_refresh:
            mock_refresh.return_value = "new_token"
            with patch.object(codex_auth, "_load"):
                result = await codex_auth.get_access_token()

        assert result == "new_token"
        mock_refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_existing_token_when_fresh(self, codex_auth: CodexAuth) -> None:
        """Should return existing token when not stale and not expired."""
        one_day_ago = int(time.time()) - (24 * 60 * 60)
        codex_auth._auth_data = {
            "access": "current_token",
            "refresh": "refresh_token",
            "expires": int(time.time()) + 3600,  # Expires in 1 hour
            "last_refresh": one_day_ago,  # 1 day ago (not stale)
        }

        with patch.object(codex_auth, "_refresh_token", new_callable=AsyncMock) as mock_refresh:
            with patch.object(codex_auth, "_load"):
                result = await codex_auth.get_access_token()

        assert result == "current_token"
        mock_refresh.assert_not_called()

    @pytest.mark.asyncio
    async def test_refreshes_when_no_access_token(self, codex_auth: CodexAuth) -> None:
        """Should refresh when access token is missing."""
        codex_auth._auth_data = {
            "refresh": "refresh_token",
            "expires": int(time.time()) + 3600,
            "last_refresh": int(time.time()),
        }

        with patch.object(codex_auth, "_refresh_token", new_callable=AsyncMock) as mock_refresh:
            mock_refresh.return_value = "new_token"
            with patch.object(codex_auth, "_load"):
                result = await codex_auth.get_access_token()

        assert result == "new_token"
        mock_refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_error_when_no_auth_data(self, codex_auth: CodexAuth) -> None:
        """Should raise 401 when no auth data exists."""
        codex_auth._auth_data = {}

        from fastapi import HTTPException

        with patch.object(codex_auth, "_load"):
            with pytest.raises(HTTPException) as exc_info:
                await codex_auth.get_access_token()

        assert exc_info.value.status_code == 401
        assert "No Codex auth data found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_raises_error_when_no_refresh_token(self, codex_auth: CodexAuth) -> None:
        """Should raise 401 when refresh is needed but no refresh token exists."""
        eight_days_ago = int(time.time()) - (8 * 24 * 60 * 60)
        codex_auth._auth_data = {
            "access": "old_token",
            "expires": int(time.time()) - 10,
            "last_refresh": eight_days_ago,
            # No "refresh" key
        }

        from fastapi import HTTPException

        with patch.object(codex_auth, "_load"):
            with pytest.raises(HTTPException) as exc_info:
                await codex_auth.get_access_token()

        assert exc_info.value.status_code == 401
        assert "No refresh token available" in exc_info.value.detail


class TestLastRefreshTracking:
    """Tests that last_refresh is properly tracked."""

    def test_last_refresh_set_on_token_exchange(self, codex_auth: CodexAuth) -> None:
        """_store_token_data should record last_refresh timestamp."""
        before = int(time.time())
        codex_auth._store_token_data({
            "access_token": "new_token",
            "refresh_token": "new_refresh",
            "expires_in": 3600,
        })
        after = int(time.time())

        assert "last_refresh" in codex_auth._auth_data
        assert before <= codex_auth._auth_data["last_refresh"] <= after

    @pytest.mark.asyncio
    async def test_last_refresh_updated_on_refresh(self, codex_auth: CodexAuth, tmp_path: Path) -> None:
        """last_refresh should be updated when token is refreshed."""
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{}")

        old_time = int(time.time()) - (8 * 24 * 60 * 60)  # 8 days ago

        with patch("anthropic_proxy.config_manager.DEFAULT_AUTH_FILE", auth_file):
            codex_auth._auth_data = {
                "access": "old_token",
                "refresh": "refresh_token",
                "expires": int(time.time()) - 10,
                "last_refresh": old_time,
            }
            codex_auth._save()

            # Mock the HTTP response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "expires_in": 3600,
            }
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                before_refresh = int(time.time())
                await codex_auth.get_access_token()
                after_refresh = int(time.time())

            # Reload to verify saved data
            codex_auth._load()
            assert codex_auth._auth_data["last_refresh"] > old_time
            assert before_refresh <= codex_auth._auth_data["last_refresh"] <= after_refresh


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing auth data."""

    def test_treats_missing_last_refresh_as_stale(self, codex_auth: CodexAuth) -> None:
        """Old auth data without last_refresh should be treated as stale."""
        # Simulate old auth data format (before this feature)
        codex_auth._auth_data = {
            "access": "token",
            "refresh": "refresh",
            "expires": int(time.time()) + 3600,
            # No last_refresh
        }
        assert codex_auth._is_token_stale() is True

    @pytest.mark.asyncio
    async def test_refreshes_legacy_auth_data(self, codex_auth: CodexAuth) -> None:
        """Should refresh when encountering legacy auth data without last_refresh."""
        codex_auth._auth_data = {
            "access": "old_token",
            "refresh": "refresh_token",
            "expires": int(time.time()) + 3600,  # Not expired
            # No last_refresh - treated as stale
        }

        with patch.object(codex_auth, "_refresh_token", new_callable=AsyncMock) as mock_refresh:
            mock_refresh.return_value = "new_token"
            with patch.object(codex_auth, "_load"):
                result = await codex_auth.get_access_token()

        assert result == "new_token"
        mock_refresh.assert_called_once()
