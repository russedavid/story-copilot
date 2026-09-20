# Optional live audio

Manual conversation entry works without audio dependencies. On macOS, install `pip install -e '.[mac-audio]'` to use microphone and system capture. Open a session's Audio controls to obtain its explicit capture command. Start only the channels you intend to record; Ctrl-C stops the capture and saves the final partial chunk. macOS may request microphone or system-audio permissions.

On the computer processing audio, install `pip install -e '.[asr]'`. The speech models must be downloaded locally and any gated model's access terms completed for that environment. Run a worker against the same private queue shown by the UI:

```sh
python -m story_copilot.audio_worker --queue /path/to/private/audio-queue --device cuda --device-index 1 --preload
```

`--preload` loads the speech models while idle; it does not start capture or playback. On a single GPU, choose index 0 and budget memory alongside the text model. CPU mode is available but may be substantially slower. The queue, model files, generated transcripts, and timing reports belong outside the repository.

For a remote worker, the capture command supports `--ssh`, `--remote-queue`, `--remote-python`, and an optional `--ssh-control` socket. The worker environment needs this package installed. Transfers are acknowledged after an idempotent import; a connection interruption retains the local spool. The UI can follow completed chunks or import them explicitly.

Microphone and system channels remain distinct. Diarization labels do not identify people by themselves: map participants to speakers and characters in the campaign. Low-confidence and overlapping speech stays available for review. Different source clocks are not silently treated as a perfectly ordered conversation.

Playback happens only when you press an audio control. Starting the UI, opening a session, or running an idle speech worker does not play or record sound.
