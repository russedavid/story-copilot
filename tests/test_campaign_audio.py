"""Inspect retained WAV bytes and authorization; never start audio playback."""

import html
import re
import time

import pytest
from starlette.testclient import TestClient

from story_copilot.campaign_web import campaign_audio_clip, wav_response
from story_copilot.campaigns import Campaigns
from story_copilot.live_audio import AudioQueue, RATE
from story_copilot.store import Store
from story_copilot.web import create_app


def recorded(store):
    c = Campaigns(store)
    cid = c.create("Synthetic audio fixture")
    sid = c.create_session(cid, "First", proactive=False)
    q = AudioQueue(store.home / "live-audio")
    chunk = q.enqueue_pcm(
        "separate-audio-session",
        "mic",
        0,
        10,
        b"\0\0" * RATE * 4,
        source={"kind": "silent_fixture"},
    )
    claim = q.claim()
    q.complete(claim, {"segments": []})
    clip = q.audio_clip(chunk, "separate-audio-session")
    mid = c.add_message(
        sid,
        "mic:0",
        "Synthetic text, not acoustic gold.",
        visibility="private",
        source={
            "kind": "live_audio",
            "chunk_id": chunk,
            "channel": "mic",
            "audio_sha256": clip["audio_sha256"],
            "start": 11.5,
            "end": 12.0,
            "transcription": {"session": "separate-audio-session"},
            "recording": {"path": "/never/read/arbitrary/source.wav"},
        },
    )
    return c, cid, sid, mid, q, chunk


def test_retained_audio_uses_chunk_offset_and_verified_branch_ancestry(tmp_path):
    c, cid, sid, mid, q, chunk = recorded(Store(tmp_path / "private"))
    clip = campaign_audio_clip(c, q, sid, mid)
    assert clip["wav_bytes"][:4] == b"RIFF"
    assert clip["seek_start"] == 1.5 and clip["seek_end"] == 2.0
    other = c.create_session(cid, "Other")
    with pytest.raises(ValueError, match="attached"):
        campaign_audio_clip(c, q, other, mid)
    branch = c.branch(sid, "Alternative")
    inherited = c.messages(branch)[0]
    assert (
        campaign_audio_clip(c, q, branch, inherited["id"])["wav_bytes"]
        == clip["wav_bytes"]
    )
    q.acknowledge("fixture", chunk)
    q.prune_audio("fixture", before=time.time() + 1)
    with pytest.raises(FileNotFoundError):
        campaign_audio_clip(c, q, sid, mid)


def test_wav_byte_ranges_are_bounded():
    data = b"RIFF" + b"\0" * 100
    response = wav_response(data, "bytes=0-43")
    assert response.status_code == 206 and response.body == data[:44]
    assert response.headers["content-range"] == "bytes 0-43/104"
    assert wav_response(data, "bytes=-4").body == data[-4:]
    for bad in ["bytes=900-", "bytes=10-2", "bytes=-0", "bytes=0-1,3-4", "items=0-2"]:
        assert wav_response(data, bad).status_code == 416


def test_source_audio_needs_explicit_private_click_and_never_autoplays(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "story_copilot.copilot.make_copilot",
        lambda store: (lambda snapshot: snapshot, None),
    )
    store = Store(tmp_path / "private")
    c, cid, sid, mid, q, chunk = recorded(store)
    app = create_app(store)
    with TestClient(app) as client:
        page = client.get(f"/play/{sid}")
        assert "Open source audio" in page.text and "<audio " not in page.text
        public = client.get(f"/play/{sid}/public")
        assert "Open source audio" not in public.text and chunk not in public.text
        control = re.search(r'<input[^>]*name="csrf_token"[^>]*>', page.text)[0]
        csrf = re.search(r'value="([^"]+)"', control)[1]
        assert client.get(f"/play/{sid}/message/{mid}/audio").status_code == 403
        opened = client.post(
            f"/play/{sid}/message/{mid}/audio-controls",
            data={"csrf_token": csrf},
            headers={"origin": "null", "sec-fetch-site": "same-origin"},
        )
        assert 'preload="none"' in opened.text and "autoplay" not in opened.text
        url = html.unescape(re.search(r'<audio[^>]*src="([^"]+)"', opened.text)[1])
        assert "#t=1.5,2.0" in url
        response = client.get(
            url, headers={"range": "bytes=0-43", "sec-fetch-site": "same-origin"}
        )
        assert response.status_code == 206 and response.content[:4] == b"RIFF"
        assert response.headers["cache-control"] == "private, no-store"
        with TestClient(app) as stranger:
            assert stranger.get(url).status_code == 403
        assert (
            client.get(
                url,
                headers={
                    "origin": "https://elsewhere.invalid",
                    "sec-fetch-site": "cross-site",
                },
            ).status_code
            == 404
        )
        q.acknowledge("fixture", chunk)
        q.prune_audio("fixture", before=time.time() + 1)
        assert client.get(url).status_code == 410
