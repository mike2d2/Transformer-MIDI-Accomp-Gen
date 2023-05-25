import pathlib
import os
import copy

from tqdm import tqdm

import mido

class Splicer(object):

    def __init__(self, file_path: str) -> None:

        self.accepted_types = ['set_tempo', 'note_on', 'note_off']

        # File path info
        self.original_path: pathlib.Path = pathlib.Path(file_path)
        self.parent_path_str: str = str(self.original_path.parent)
        self.file_name: str = self.original_path.parts[-1][:-4]
        parts_temp: list[str] = list(self.original_path.parent.parts)
        data_ind: int = parts_temp.index('data_raw')
        parts_temp[data_ind] = 'data_spliced'
        self.new_dir: pathlib.Path = pathlib.Path('/'.join(parts_temp))

        # Make merged mido file
        self.mido_obj: mido.MidiFile = mido.MidiFile(self.original_path)
        self.mido_obj.tracks = [mido.merge_tracks(self.mido_obj.tracks)]

        # Extract program change tracks
        self.program_track: list[mido.Message] = [msg for msg in self.mido_obj.tracks[0] if msg.type == 'program_change']
        for i in range(len(self.program_track)):
            self.program_track[i].time = 0

        # Extract ticks per beat of the song
        self.ticks_per_beat: int = self.mido_obj.ticks_per_beat

        # Extract time signature events
        self.time_sigs: list[mido.MetaMessage] = [msg for msg in self.mido_obj.tracks[0] if msg.type == 'time_signature']
        for i in range(len(self.time_sigs)):
            self.time_sigs[i].time = 0

        # Extract a default tempo
        self.default_tempo: int = 0
        for msg in self.mido_obj.tracks[0]:
            if msg.type == 'set_tempo':
                self.default_tempo = msg.tempo
                break

    def write_spliced(self, max_seconds: int) -> None:
        '''Given number of seconds for clips and for overlap duration, write clips from loaded file.'''

        # Initialize loop trackers
        end: int = max_seconds
        i: int = 0
        track_time: float = 0
        current_tempo: None | mido.MetaMessage = None

        # Main loop over messages
        k: int = 0
        while end < self.mido_obj.length:
            
            # Construct spliced track
            clip: mido.MidiFile = mido.MidiFile()
            clip.ticks_per_beat = self.ticks_per_beat
            new_track: list = copy.deepcopy(self.program_track)
            for msg in self.time_sigs:
                new_track.append(msg)
            if current_tempo:
                new_track.append(current_tempo)
            for msg in self.mido_obj.tracks[0][k:]:
                used_tempo: int = current_tempo.tempo if current_tempo else self.default_tempo
                increment: float = mido.tick2second(msg.time, self.ticks_per_beat, used_tempo)
                if track_time + increment > end:
                    break
                if msg.type in self.accepted_types:
                    track_time += increment
                    if msg.type == 'set_tempo':
                        msg_zero_time: mido.MetaMessage = msg
                        msg_zero_time.time = 0
                        current_tempo = msg_zero_time
                    new_track.append(msg)
                    k += 1
                else:
                    k += 1
                    continue
            clip.tracks = [new_track]
            
            # Save new file
            if not os.path.exists(self.new_dir):
                os.makedirs(self.new_dir)
            clip.save(self.new_dir / f'{self.file_name}_sec{max_seconds}_{i}.mid')
            end += max_seconds
            i += 1

    def check_valid(self) -> bool:
        '''Check if the track has tempo information, tick information, and tracks.'''

        if any([not self.ticks_per_beat, not self.default_tempo, not self.program_track]):
            return False
        return True


class DatasetSplicer(object):

    def __init__(self, data_directory: str) -> None:
        
        self.file_paths: map = map(
            str,
            pathlib.Path(data_directory).rglob('*.mid')
        )

    def write_spliced(self, seconds: int, cap: int = 0) -> None:
        '''Given a tokenizer and a directory of midi files, write cleaned data to given output'''

        # Iterate through dir
        completed_dirs: list[str] = []
        count: int = 0
        for file in tqdm(self.file_paths):

            # Load splicer
            try:
                splicer_loaded = Splicer(file)
            except:
                continue

            # Skip if type 2
            if splicer_loaded.mido_obj.type == 2:
                continue

            # Check if we have already taken a song from this dir
            if splicer_loaded.parent_path_str in completed_dirs:
                continue
            else:
                completed_dirs.append(splicer_loaded.parent_path_str)

            # Check validity
            if not splicer_loaded.check_valid():
                continue

            # Write spliced
            splicer_loaded.write_spliced(seconds)

            # If capped check if we should finish
            if cap:
                count += 1
                print(count)
                if count >= cap:
                    break


if __name__ == '__main__':
    
    cap = 2000
    seconds = 30
    data_dir = '../data_raw/'

    dataset_splicer = DatasetSplicer(data_dir)
    dataset_splicer.write_spliced(seconds, cap)
