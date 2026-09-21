"""Create file-only synthetic speech fixtures on macOS; never play or capture audio."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

from .workflow_fixtures import SCENES


def prepare(output, *, facilitator_voice="Daniel", player_voice="Samantha"):
    output=Path(output).expanduser().resolve()
    if output.exists() or any((p/".git").exists() for p in [output,*output.parents]):
        raise ValueError("Choose a new fixture directory outside repositories.")
    if not Path('/usr/bin/say').is_file() or not shutil.which('ffmpeg'):
        raise ValueError("This optional fixture generator requires macOS speech synthesis and ffmpeg.")
    output.mkdir(parents=True,mode=0o700)
    manifest={"kind":"authored_synthetic_audio","capture":False,"playback":False,"sample_rate":16000,"scenes":[]}
    clock=0.0
    for i,scene in enumerate(SCENES):
        aiff=output/f'{i:02d}-{scene["id"]}.aiff';wav=output/f'{i:02d}-{scene["id"]}.wav'
        voice=facilitator_voice if scene['channel']=='mic' else player_voice
        subprocess.run(['/usr/bin/say','-v',voice,'-r','170','-o',str(aiff),scene['text']],check=True)
        subprocess.run(['ffmpeg','-v','error','-i',str(aiff),'-ac','1','-ar','16000','-c:a','pcm_s16le',str(wav)],check=True)
        with wave.open(str(wav),'rb') as source:
            frames=source.getnframes();pcm=source.readframes(frames)
        duration=frames/16000
        manifest['scenes'].append({**scene,"file":wav.name,"voice":voice,"start":round(clock,5),"duration":duration,"pcm_sha256":hashlib.sha256(pcm).hexdigest()})
        clock+=duration+1.0
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    manifest['recording_id']=identity
    manifest['clock_note']='Known synthetic source schedule shared by both channels; not a calendar or a live capture.'
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args();m=prepare(args.output);print(json.dumps({'scenes':len(m['scenes']),'duration':sum(s['duration'] for s in m['scenes'])}))
