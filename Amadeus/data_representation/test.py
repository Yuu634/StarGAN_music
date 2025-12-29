"""
corpusファイルを読み込んで音符の属性を詳細に出力

使用方法:
    python analyze_corpus.py --input ../dataset/represented_data/corpus/corput_test/sample.pkl
"""

import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any


class CorpusAnalyzer:
    """Corpusデータを解析するクラス"""
    
    def __init__(self, corpus_path: str):
        """
        Parameters:
        -----------
        corpus_path : str
            corpus.pklファイルのパス
        """
        self.corpus_path = corpus_path
        self.corpus = self._load_corpus(corpus_path)
    
    def _load_corpus(self, corpus_path: str) -> Dict:
        """corpusファイルを読み込み"""
        with open(corpus_path, 'rb') as f:
            corpus = pickle.load(f)
        return corpus
    
    def print_structure(self):
        """corpusの全体構造を表示"""
        print(f"\n{'='*80}")
        print(f"Corpus Structure: {Path(self.corpus_path).name}")
        print(f"{'='*80}\n")
        
        print("Top-level keys:")
        for key in self.corpus.keys():
            print(f"  • {key}")
        print()
    
    def print_metadata(self):
        """メタデータを表示"""
        print(f"{'='*80}")
        print("Metadata")
        print(f"{'='*80}\n")
        
        metadata = self.corpus.get('metadata', {})
        
        for key, value in metadata.items():
            if key == 'time_signature':
                print(f"  {key}:")
                for ts in value:
                    if isinstance(ts, tuple) and len(ts) == 3:
                        time, num, denom = ts
                        print(f"    Time {time}: {num}/{denom}")
                    else:
                        print(f"    {ts}")
            else:
                print(f"  {key}: {value}")
        print()
    
    def print_notes_summary(self):
        """音符の統計情報を表示"""
        print(f"{'='*80}")
        print("Notes Summary")
        print(f"{'='*80}\n")
        
        notes = self.corpus.get('notes', {})
        
        total_notes = 0
        pitch_range = [127, 0]  # [min, max]
        duration_range = [float('inf'), 0]
        velocity_range = [127, 0]
        
        for instr, instr_notes in notes.items():
            instr_total = sum(len(note_list) for note_list in instr_notes.values())
            total_notes += instr_total
            
            print(f"Instrument {instr}:")
            print(f"  • Total notes: {instr_total}")
            print(f"  • Time positions: {len(instr_notes)}")
            
            # 音符の統計
            for time, note_list in instr_notes.items():
                for note in note_list:
                    pitch_range[0] = min(pitch_range[0], note.pitch)
                    pitch_range[1] = max(pitch_range[1], note.pitch)
                    
                    if hasattr(note, 'quantized_duration'):
                        duration = note.quantized_duration
                    else:
                        duration = note.end - note.start
                    duration_range[0] = min(duration_range[0], duration)
                    duration_range[1] = max(duration_range[1], duration)
                    
                    velocity_range[0] = min(velocity_range[0], note.velocity)
                    velocity_range[1] = max(velocity_range[1], note.velocity)
            print()
        
        print(f"Total notes across all instruments: {total_notes}")
        print(f"Pitch range: {pitch_range[0]} - {pitch_range[1]}")
        print(f"Duration range: {duration_range[0]} - {duration_range[1]}")
        print(f"Velocity range: {velocity_range[0]} - {velocity_range[1]}")
        print()
    
    def print_all_notes_detailed(self, max_notes: int = 100):
        """全音符の詳細属性を表示"""
        print(f"{'='*80}")
        print(f"Detailed Note Attributes (showing first {max_notes} notes)")
        print(f"{'='*80}\n")
        
        notes = self.corpus.get('notes', {})
        note_count = 0
        
        for instr, instr_notes in notes.items():
            print(f"\n{'─'*80}")
            print(f"Instrument {instr}")
            print(f"{'─'*80}\n")
            
            # 時刻順にソート
            sorted_times = sorted(instr_notes.keys())
            
            for time in sorted_times:
                note_list = instr_notes[time]
                
                print(f"  Time: {time} ticks")
                print(f"  {'─'*76}")
                
                for i, note in enumerate(note_list):
                    if note_count >= max_notes:
                        print(f"\n  ... (showing only first {max_notes} notes)")
                        return
                    
                    # 音符属性の表示
                    print(f"    Note {note_count + 1}:")
                    print(f"      • Pitch: {note.pitch} ({self._pitch_to_name(note.pitch)})")
                    print(f"      • Start: {note.start} ticks")
                    print(f"      • End: {note.end} ticks")
                    print(f"      • Duration: {note.end - note.start} ticks")
                    
                    if hasattr(note, 'quantized_duration'):
                        print(f"      • Quantized Duration: {note.quantized_duration}")
                    
                    print(f"      • Velocity: {note.velocity}")
                    
                    # その他の属性があれば表示
                    for attr in dir(note):
                        if not attr.startswith('_') and attr not in ['pitch', 'start', 'end', 'velocity', 'quantized_duration']:
                            value = getattr(note, attr)
                            if not callable(value):
                                print(f"      • {attr}: {value}")
                    
                    print()
                    note_count += 1
                
                print()
    
    def print_chords(self):
        """コード情報を表示"""
        print(f"{'='*80}")
        print("Chord Information")
        print(f"{'='*80}\n")
        
        chords = self.corpus.get('chords', {})
        
        if not chords:
            print("  No chord information available\n")
            return
        
        sorted_times = sorted(chords.keys())
        
        for time in sorted_times[:20]:  # 最初の20個
            chord_list = chords[time]
            print(f"  Time {time}:")
            for chord in chord_list:
                if hasattr(chord, 'text'):
                    print(f"    • {chord.text}")
                else:
                    print(f"    • {chord}")
        
        if len(sorted_times) > 20:
            print(f"\n  ... ({len(sorted_times) - 20} more chord positions)")
        print()
    
    def print_tempos(self):
        """テンポ情報を表示"""
        print(f"{'='*80}")
        print("Tempo Changes")
        print(f"{'='*80}\n")
        
        tempos = self.corpus.get('tempos', {})
        
        if not tempos:
            print("  No tempo information available\n")
            return
        
        sorted_times = sorted(tempos.keys())
        
        for time in sorted_times:
            tempo_list = tempos[time]
            print(f"  Time {time}:")
            for tempo in tempo_list:
                if hasattr(tempo, 'tempo'):
                    print(f"    • {tempo.tempo} BPM")
                else:
                    print(f"    • {tempo} BPM")
        print()
    
    def _pitch_to_name(self, pitch: int) -> str:
        """MIDIピッチ番号を音名に変換"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (pitch // 12) - 1
        note = note_names[pitch % 12]
        return f"{note}{octave}"
    
    def export_to_csv(self, output_path: str):
        """音符データをCSVに出力"""
        import csv
        
        print(f"\n{'='*80}")
        print(f"Exporting to CSV: {output_path}")
        print(f"{'='*80}\n")
        
        notes = self.corpus.get('notes', {})
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # ヘッダー
            writer.writerow([
                'Instrument', 'Time', 'Pitch', 'Pitch_Name', 
                'Start', 'End', 'Duration', 'Quantized_Duration', 'Velocity'
            ])
            
            # データ
            for instr, instr_notes in notes.items():
                for time, note_list in sorted(instr_notes.items()):
                    for note in note_list:
                        quantized_dur = getattr(note, 'quantized_duration', '')
                        
                        writer.writerow([
                            instr,
                            time,
                            note.pitch,
                            self._pitch_to_name(note.pitch),
                            note.start,
                            note.end,
                            note.end - note.start,
                            quantized_dur,
                            note.velocity
                        ])
        
        print(f"✓ CSV exported successfully\n")
    
    def analyze_all(self, max_notes: int = 100, export_csv: bool = False):
        """全ての解析を実行"""
        self.print_structure()
        self.print_metadata()
        self.print_notes_summary()
        self.print_chords()
        self.print_tempos()
        self.print_all_notes_detailed(max_notes=max_notes)
        
        if export_csv:
            csv_path = str(Path(self.corpus_path).with_suffix('.csv'))
            self.export_to_csv(csv_path)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze corpus.pkl file and print note attributes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_corpus.py --input ../dataset/represented_data/corpus/corput_test/sample.pkl
  
  # Show more notes
  python analyze_corpus.py --input sample.pkl --max_notes 200
  
  # Export to CSV
  python analyze_corpus.py --input sample.pkl --export_csv
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to corpus.pkl file')
    parser.add_argument('--max_notes', type=int, default=100,
                        help='Maximum number of notes to display in detail (default: 100)')
    parser.add_argument('--export_csv', action='store_true',
                        help='Export notes to CSV file')
    parser.add_argument('--structure_only', action='store_true',
                        help='Show only structure and summary')
    
    args = parser.parse_args()
    
    # 解析実行
    analyzer = CorpusAnalyzer(corpus_path=args.input)
    
    if args.structure_only:
        analyzer.print_structure()
        analyzer.print_metadata()
        analyzer.print_notes_summary()
    else:
        analyzer.analyze_all(
            max_notes=args.max_notes,
            export_csv=args.export_csv
        )


if __name__ == "__main__":
    main()