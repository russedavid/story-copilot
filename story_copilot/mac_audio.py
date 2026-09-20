"""Explicit microphone/system capture into the private audio queue.

The audio-only Core Audio tap is adapted from the same author's Over The Shoulder
Coder (https://github.com/russedavid/over-the-shoulder). No display capture or
ScreenCaptureKit is used. Optional dependencies are sounddevice, numpy, scipy,
and (for macOS system audio) pyobjc-framework-CoreAudio. Nothing starts on import.
"""

import ctypes as ct
import math
import os
import platform
import threading
import time
from collections import deque
from pathlib import Path
from uuid import uuid4


class PropertyAddress(ct.Structure):
    _fields_ = [
        ("selector", ct.c_uint32),
        ("scope", ct.c_uint32),
        ("element", ct.c_uint32),
    ]


class StreamFormat(ct.Structure):
    _fields_ = [
        ("rate", ct.c_double),
        ("format", ct.c_uint32),
        ("flags", ct.c_uint32),
        ("bytes_per_packet", ct.c_uint32),
        ("frames_per_packet", ct.c_uint32),
        ("bytes_per_frame", ct.c_uint32),
        ("channels", ct.c_uint32),
        ("bits", ct.c_uint32),
        ("reserved", ct.c_uint32),
    ]


class AudioTimeStamp(ct.Structure):
    _fields_ = [
        ("sample_time", ct.c_double),
        ("host_time", ct.c_uint64),
        ("rate_scalar", ct.c_double),
        ("word_clock", ct.c_uint64),
        ("smpte", ct.c_ubyte * 24),
        ("flags", ct.c_uint32),
        ("reserved", ct.c_uint32),
    ]


class AudioBuffer(ct.Structure):
    _fields_ = [("channels", ct.c_uint32), ("size", ct.c_uint32), ("data", ct.c_void_p)]


class AudioBuffers(ct.Structure):
    _fields_ = [("count", ct.c_uint32), ("buffers", AudioBuffer * 1)]


IO_CALLBACK = ct.CFUNCTYPE(
    ct.c_int32,
    ct.c_uint32,
    ct.c_void_p,
    ct.POINTER(AudioBuffers),
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
)


def decode_buffers(pointer, format):
    import numpy as np

    if not pointer or not 0 < pointer.contents.count <= 32:
        return np.array([], dtype=np.float32)
    endian = ">" if format.flags & 2 else "<"
    if format.flags & 1 and format.bits in {32, 64}:
        dtype, scale = np.dtype(endian + f"f{format.bits // 8}"), 1.0
    elif format.flags & 4 and format.bits in {16, 32}:
        dtype, scale = (
            np.dtype(endian + f"i{format.bits // 8}"),
            float(2 ** (format.bits - 1)),
        )
    else:
        raise ValueError("Unsupported system-audio PCM format")
    buffers = (AudioBuffer * pointer.contents.count).from_address(
        ct.addressof(pointer.contents) + AudioBuffers.buffers.offset
    )
    channels = []
    for buffer in buffers:
        if not buffer.data or not buffer.size or not buffer.channels:
            continue
        if buffer.size > 8_000_000 or buffer.channels > 32:
            raise ValueError("Unexpected system-audio buffer size")
        samples = (
            np.frombuffer(ct.string_at(buffer.data, buffer.size), dtype=dtype).astype(
                np.float32
            )
            / scale
        )
        count = len(samples) // buffer.channels
        if count:
            channels.extend(
                samples[: count * buffer.channels].reshape(count, buffer.channels).T
            )
    if not channels:
        return np.array([], dtype=np.float32)
    count = min(map(len, channels))
    return np.mean(np.stack([channel[:count] for channel in channels]), axis=0).astype(
        np.float32
    )


def coreaudio_timestamp(pointer, convert_host_time, *, fallback, frames, rate):
    stamp = ct.cast(pointer, ct.POINTER(AudioTimeStamp)).contents if pointer else None
    if stamp is not None and stamp.flags & 2:
        return convert_host_time(stamp.host_time) / 1e9, "CoreAudio host timestamp"
    return (
        fallback - frames / rate,
        "callback arrival minus duration; device timestamp unavailable",
    )


def portaudio_timestamp(timing, *, fallback, frames, rate):
    try:
        adc, current = float(timing.inputBufferAdcTime), float(timing.currentTime)
        if (
            math.isfinite(adc)
            and math.isfinite(current)
            and (adc or current)
            and abs(adc - current) < 5
        ):
            return (
                fallback + adc - current,
                "PortAudio ADC timestamp mapped by callback current time; approximate host offset",
            )
    except (AttributeError, TypeError, ValueError):
        pass
    return (
        fallback - frames / rate,
        "callback arrival minus duration; device timestamp unavailable",
    )


class SystemAudioTap:
    def __init__(self):
        self.tap_id = self.device_id = 0
        self.io_id = ct.c_void_p()
        self.callback = None
        self.format = None
        self.chunks = deque()
        self.clock_chunks = deque()
        self.sample_count = 0
        self.lock = threading.Lock()
        self.closed = False
        self.error = None
        self.native = None
        self.started = False

    def _check(self, status, operation):
        if status:
            raise RuntimeError(
                f"System audio {operation} failed ({status}). Check macOS System Audio Recording permission for the launcher."
            )

    def _property(self, object_id, selector, value, qualifier=None):
        import CoreAudio as C

        address = PropertyAddress(selector, C.kAudioObjectPropertyScopeGlobal, 0)
        size = ct.c_uint32(ct.sizeof(value))
        self._check(
            self.native.AudioObjectGetPropertyData(
                object_id,
                ct.byref(address),
                ct.sizeof(qualifier) if qualifier is not None else 0,
                ct.byref(qualifier) if qualifier is not None else None,
                ct.byref(size),
                ct.byref(value),
            ),
            "property lookup",
        )
        return value

    def start(self):
        if self.started or self.closed:
            raise RuntimeError("Create a fresh tap for a new capture session.")
        import CoreAudio as C

        version = tuple(int(n) for n in platform.mac_ver()[0].split(".")[:2])
        if version < (14, 2):
            raise RuntimeError(
                "Audio-only system capture requires macOS 14.2 or newer; microphone-only capture remains available on older systems"
            )
        self.native = ct.CDLL(
            "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
        )
        self.native.AudioObjectGetPropertyData.argtypes = [
            ct.c_uint32,
            ct.POINTER(PropertyAddress),
            ct.c_uint32,
            ct.c_void_p,
            ct.POINTER(ct.c_uint32),
            ct.c_void_p,
        ]
        self.native.AudioDeviceCreateIOProcID.argtypes = [
            ct.c_uint32,
            IO_CALLBACK,
            ct.c_void_p,
            ct.POINTER(ct.c_void_p),
        ]
        self.native.AudioDeviceStart.argtypes = [ct.c_uint32, ct.c_void_p]
        self.native.AudioDeviceStop.argtypes = [ct.c_uint32, ct.c_void_p]
        self.native.AudioDeviceDestroyIOProcID.argtypes = [ct.c_uint32, ct.c_void_p]
        excluded = []
        try:
            process = self._property(
                C.kAudioObjectSystemObject,
                C.kAudioHardwarePropertyTranslatePIDToProcessObject,
                ct.c_uint32(),
                ct.c_int32(os.getpid()),
            ).value
            if process:
                excluded.append(process)
        except RuntimeError:
            pass  # A process with no audio I/O may not yet have a HAL object.
        description = C.CATapDescription.alloc().initMonoGlobalTapButExcludeProcesses_(
            excluded
        )
        description.setName_("Story Copilot system audio")
        description.setPrivate_(True)
        description.setMuteBehavior_(C.CATapUnmuted)
        try:
            status, self.tap_id = C.AudioHardwareCreateProcessTap(description, None)
            self._check(status, "tap creation")
            self.format = self._property(
                self.tap_id, C.kAudioTapPropertyFormat, StreamFormat()
            )
            if (
                self.format.format != int.from_bytes(b"lpcm", "big")
                or not 8000 <= self.format.rate <= 192000
            ):
                raise ValueError("System audio returned an unsupported stream format")
            spec = {
                # These SDK constants are C strings (bytes in PyObjC). Passing
                # bytes would bridge CFData, but HAL expects CFString keys.
                C.kAudioAggregateDeviceNameKey.decode(): "Story Copilot private audio input",
                C.kAudioAggregateDeviceUIDKey.decode(): "facilitator-" + uuid4().hex,
                C.kAudioAggregateDeviceIsPrivateKey.decode(): True,
                C.kAudioAggregateDeviceTapAutoStartKey.decode(): True,
                C.kAudioAggregateDeviceTapListKey.decode(): [
                    {
                        C.kAudioSubTapUIDKey.decode(): str(
                            description.UUID().UUIDString()
                        ),
                        C.kAudioSubTapDriftCompensationKey.decode(): True,
                    }
                ],
            }
            status, self.device_id = C.AudioHardwareCreateAggregateDevice(spec, None)
            self._check(status, "private input creation")

            def receive(device, now, data, input_time, output, output_time, context):
                try:
                    if not self.closed:
                        samples = decode_buffers(data, self.format)
                        if samples.size:
                            with self.lock:
                                packet_start, quality = coreaudio_timestamp(
                                    input_time,
                                    C.AudioConvertHostTimeToNanos,
                                    fallback=time.monotonic(),
                                    frames=samples.size,
                                    rate=self.format.rate,
                                )
                                self.chunks.append(samples)
                                self.clock_chunks.append((packet_start, quality))
                                self.sample_count += samples.size
                                if self.sample_count > self.format.rate * 5:
                                    self.error = "Capture buffer overflow; pause and process the backlog"
                                    self.closed = True
                except Exception as error:
                    self.error = f"{type(error).__name__}: {error}"
                return 0

            self.callback = IO_CALLBACK(receive)
            self._check(
                self.native.AudioDeviceCreateIOProcID(
                    self.device_id, self.callback, None, ct.byref(self.io_id)
                ),
                "callback creation",
            )
            self._check(
                self.native.AudioDeviceStart(self.device_id, self.io_id), "start"
            )
            self.started = True
        except Exception:
            self.stop()
            raise

    def drain_timed(self):
        import numpy as np
        from scipy.signal import resample_poly

        if self.error:
            raise RuntimeError("System-audio decoding failed: " + self.error)
        with self.lock:
            chunks = list(self.chunks)
            clocks = list(self.clock_chunks)
            self.chunks.clear()
            self.clock_chunks.clear()
            self.sample_count = 0
        if not chunks:
            return []
        groups = []
        rate = int(round(self.format.rate))
        for index, samples in enumerate(chunks):
            stamp, quality = (
                clocks[index] if index < len(clocks) else (None, "unlocated buffer")
            )
            if (
                groups
                and stamp is not None
                and groups[-1]["start_monotonic"] is not None
                and quality == groups[-1]["quality"]
                and abs(
                    stamp
                    - (groups[-1]["start_monotonic"] + groups[-1]["frames"] / rate)
                )
                < 2 / rate
            ):
                groups[-1]["samples"].append(samples)
                groups[-1]["frames"] += len(samples)
            else:
                groups.append(
                    {
                        "samples": [samples],
                        "frames": len(samples),
                        "start_monotonic": stamp,
                        "quality": quality,
                    }
                )
        packets = []
        for group in groups:
            samples = np.concatenate(group["samples"])
            if rate != 16000:
                divisor = math.gcd(rate, 16000)
                samples = resample_poly(samples, 16000 // divisor, rate // divisor)
                if group["start_monotonic"] is not None:
                    # Quantize both boundaries onto one absolute16kHz grid;
                    # independently rounded packets otherwise accumulate drift.
                    first = round(group["start_monotonic"] * 16000)
                    last = round(
                        (group["start_monotonic"] + group["frames"] / rate) * 16000
                    )
                    samples = samples[: last - first]
                    group["start_monotonic"] = first / 16000
            packets.append(
                {
                    "data": samples.astype(np.float32),
                    "start_monotonic": group["start_monotonic"],
                    "quality": group["quality"],
                }
            )
        return packets

    def drain(self):
        import numpy as np

        packets = self.drain_timed()
        return (
            np.concatenate([p["data"] for p in packets])
            if packets
            else np.array([], dtype=np.float32)
        )

    def stop(self):
        import CoreAudio as C

        self.closed = True
        if self.device_id and self.io_id.value:
            if self.started:
                self.native.AudioDeviceStop(self.device_id, self.io_id)
                self.started = False
            self.native.AudioDeviceDestroyIOProcID(self.device_id, self.io_id)
            self.io_id = ct.c_void_p()
        if self.device_id:
            C.AudioHardwareDestroyAggregateDevice(self.device_id)
            self.device_id = 0
        if self.tap_id:
            C.AudioHardwareDestroyProcessTap(self.tap_id)
            self.tap_id = 0
        self.callback = None


class Microphone:
    """Bounded callback buffer; device overflow is reported, never silently hidden."""

    def __init__(self, device=None):
        self.device = device
        self.stream = None
        self.buffer = bytearray()
        self.clock_packets = []
        self.lock = threading.Lock()
        self.error = None

    def start(self):
        if self.stream is not None:
            raise RuntimeError("Microphone capture is already open.")
        import sounddevice as sd

        def callback(data, frames, timing, status):
            with self.lock:
                if status:
                    self.error = f"Microphone input status: {status}"
                if len(self.buffer) + len(data) > 16000 * 2 * 5:
                    self.error = "Microphone capture buffer overflow; pause and process the backlog"
                    return
                sample_start, quality = portaudio_timestamp(
                    timing, fallback=time.monotonic(), frames=frames, rate=16000
                )
                self.buffer.extend(data)
                self.clock_packets.append(
                    {
                        "data": bytes(data),
                        "start_monotonic": sample_start,
                        "quality": quality,
                    }
                )

        self.stream = sd.RawInputStream(
            samplerate=16000,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=callback,
        )
        try:
            self.stream.start()
        except BaseException:
            stream, self.stream = self.stream, None
            stream.close()
            raise

    def drain_timed(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            packets = list(self.clock_packets)
            if not packets and self.buffer:
                packets = [
                    {
                        "data": bytes(self.buffer),
                        "start_monotonic": None,
                        "quality": "unlocated buffer",
                    }
                ]
            self.buffer.clear()
            self.clock_packets.clear()
            return packets

    def drain(self):
        return b"".join(p["data"] for p in self.drain_timed())

    def stop(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()


def capture(
    queue,
    *,
    session_id,
    mic=False,
    system=False,
    duration=None,
    chunk_seconds=30,
    overlap_seconds=2,
    microphone_device=None,
    stop=None,
    transport=None,
):
    """Opt-in blocking capture; use a thread/process owned by an explicit UI action.

    Microphone/system channels keep their own samples, speaker clusters and source
    IDs. They are not guaranteed echo-free: use headphones to avoid remote audio
    entering both sources. Full queues stop capture instead of losing speech.
    """
    import time
    from .live_audio import TimedPCMChunker, RATE

    def encode_system(data):
        import numpy as np

        return (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()

    if not mic and not system:
        raise ValueError("Explicitly enable microphone and/or system audio.")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Capture duration must be positive.")
    stop = stop or threading.Event()
    if transport is not None and hasattr(transport, "cancel"):
        transport.cancel = stop
    devices = {}
    if mic:
        devices["mic"] = Microphone(microphone_device)
    if system:
        devices["system"] = SystemAudioTap()
    capture_id = uuid4().hex
    started_at = time.time()
    started_monotonic = time.monotonic()
    # A host-boot ID prevents comparing monotonic epochs after a reboot. The
    # five-second clock pre-roll is an origin convention, not recorded silence.
    import hashlib
    import subprocess

    if platform.system() == "Darwin":
        boot = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.boottime"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    else:
        boot = str(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    common_clock = queue.capture_clock(
        session_id,
        boot_id=hashlib.sha256(boot.encode()).hexdigest(),
        utc_now=started_at - 5,
        monotonic_now=started_monotonic - 5,
    )
    with queue.db() as db:
        tails = {
            channel: db.execute(
                "SELECT sequence FROM chunks WHERE session=? AND channel=? ORDER BY sequence DESC LIMIT 1",
                (session_id, channel),
            ).fetchone()
            for channel in devices
        }
    chunkers, sources = {}, {}
    for channel, tail in tails.items():
        chunkers[channel] = TimedPCMChunker(
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            sequence=tail["sequence"] + 1 if tail else 0,
        )
        sources[channel] = {
            "kind": "live_capture",
            "capture_id": capture_id,
            "channel": channel,
            "wall_clock_started": started_at,
            "sample_rate": RATE,
            "backend": "coreaudio-process-tap"
            if channel == "system"
            else "sounddevice",
            "resumed": bool(tail),
            "clock": dict(common_clock),
            "echo_cancellation": False,
        }
    started_devices, chunks = [], []

    def preserve(channel, part):
        from .live_audio import packed, sha

        recovery = (
            queue.home / f"capture-recovery-{capture_id}-{channel}-{part['sequence']}"
        )
        recovery.with_suffix(".pcm").write_bytes(part["pcm"])
        os.chmod(recovery.with_suffix(".pcm"), 0o600)
        recovery.with_suffix(".json").write_text(
            packed(
                {
                    **{k: v for k, v in part.items() if k != "pcm"},
                    "session_id": session_id,
                    "channel": channel,
                    "source": sources[channel],
                    "pcm_file": str(recovery.with_suffix(".pcm")),
                    "pcm_sha256": sha(part["pcm"]),
                }
            )
        )
        os.chmod(recovery.with_suffix(".json"), 0o600)

    def emit(channel, data, *, stamp=None, quality=None, final=False):
        if quality:
            current = sources[channel]["clock"].get("timestamp_quality")
            sources[channel]["clock"]["timestamp_quality"] = (
                quality
                if current in {None, quality}
                else "mixed callback/device timestamp quality"
            )
        relative = (
            stamp - common_clock["origin_monotonic"] if stamp is not None else None
        )
        parts = chunkers[channel].feed(data, start_seconds=relative, final=final)
        for index, part in enumerate(parts):
            try:
                chunks.append(
                    queue.enqueue_pcm(
                        session_id, channel, source=sources[channel], **part
                    )
                )
            except Exception:
                for remaining in parts[index:]:
                    preserve(channel, remaining)
                raise

    def drain_channel(channel, device, *, final=False):
        if hasattr(device, "drain_timed"):
            packets = device.drain_timed()
        else:
            data = device.drain()
            packets = (
                [
                    {
                        "data": data,
                        "start_monotonic": None,
                        "quality": "legacy callback arrival estimate",
                    }
                ]
                if len(data)
                else []
            )
        for packet in packets:
            data = (
                encode_system(packet["data"]) if channel == "system" else packet["data"]
            )
            if not data:
                continue
            stamp = packet["start_monotonic"]
            quality = packet["quality"]
            if stamp is None:
                stamp = time.monotonic() - len(data) / (RATE * 2)
                quality = "callback arrival minus duration estimate; original device clock unavailable"
            emit(channel, data, stamp=stamp, quality=quality)
        if final:
            emit(channel, b"", final=True)

    try:
        for channel, device in devices.items():
            device.start()
            started_devices.append(device)
        started = time.monotonic()
        while not stop.is_set() and (
            duration is None or time.monotonic() - started < duration
        ):
            for channel, device in devices.items():
                drain_channel(channel, device)
                for part in chunkers[channel].flush_idle(
                    time.monotonic() - common_clock["origin_monotonic"]
                ):
                    try:
                        chunks.append(
                            queue.enqueue_pcm(
                                session_id, channel, source=sources[channel], **part
                            )
                        )
                    except Exception:
                        preserve(channel, part)
                        raise
            if transport is not None:
                try:
                    transport.transfer(queue)
                except InterruptedError:
                    if not stop.is_set():
                        raise
                    break
            stop.wait(0.1)
        # Stop the sources before draining so no tail arrives after finalization.
        for device in reversed(started_devices):
            device.stop()
        started_devices.clear()
        for channel, device in devices.items():
            drain_channel(channel, device, final=True)
        if transport is not None:
            # Hardware is already stopped. Give the saved tail one short final
            # transfer attempt, with no ongoing capture during a reconnect.
            if hasattr(transport, "cancel"):
                transport.cancel = None
                previous_timeout = transport.timeout
                transport.timeout = min(previous_timeout, 2.0)
                try:
                    transport.transfer(queue)
                finally:
                    transport.timeout = previous_timeout
            else:
                transport.transfer(queue)
        return chunks
    finally:
        for device in reversed(started_devices):
            device.stop()
        # An interrupted transfer or full spool must also preserve partial tails.
        for channel, device in devices.items():
            if chunkers[channel].closed:
                continue
            try:
                drain_channel(channel, device, final=True)
            except Exception:
                if not chunkers[channel].closed:
                    for part in chunkers[channel].feed(b"", final=True):
                        preserve(channel, part)


def main():
    import argparse
    import signal
    from .live_audio import AudioQueue, packed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--mic", action="store_true")
    parser.add_argument("--system", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--chunk-seconds", type=float, default=30)
    parser.add_argument("--microphone-device")
    parser.add_argument("--ssh")
    parser.add_argument("--remote-queue")
    parser.add_argument("--remote-python", default="python")
    parser.add_argument("--ssh-control")
    args = parser.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    print(
        "Capture explicitly enabled; Ctrl-C stops and saves the final partial chunk.",
        flush=True,
    )
    from .live_audio import SSHQueueTransport

    if bool(args.ssh) != bool(args.remote_queue):
        parser.error("Use --ssh and --remote-queue together.")
    transport = (
        SSHQueueTransport(
            args.ssh,
            args.remote_queue,
            remote_python=args.remote_python,
            control_path=args.ssh_control,
        )
        if args.ssh
        else None
    )
    try:
        result = capture(
            AudioQueue(args.queue),
            session_id=args.session,
            mic=args.mic,
            system=args.system,
            duration=args.duration,
            chunk_seconds=args.chunk_seconds,
            microphone_device=args.microphone_device,
            stop=stop,
            transport=transport,
        )
    finally:
        if transport:
            transport.close()

    print(packed({"chunks": result, "capture": "stopped"}), flush=True)


if __name__ == "__main__":
    main()
