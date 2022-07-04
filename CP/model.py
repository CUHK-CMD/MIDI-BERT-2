from tqdm import tqdm
import numpy as np
import pickle
import utils as utils
import miditoolkit


class CP(object):
    def __init__(self, dict):
        # load dictionary
        self.dict = dict
        self.event2word, self.word2event = pickle.load(open(dict, "rb"))
        # pad word: ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>']
        self.pad_word = [
            self.event2word[etype]["%s <PAD>" % etype] for etype in self.event2word
        ]

    def extract_events(self, input_path, task):
        note_items, tempo_items = utils.read_items(input_path)
        pianohist = None
        ############################################################
        if task == "custom":
            midi_obj = miditoolkit.midi.parser.MidiFile(input_path)
            # Add 'Program' to each raw token
            for i in note_items:
                i.Program = utils.Type2Program(midi_obj, i.Type)
            # Add 'TimeSignature' to each raw token
            for i in note_items:
                i.TimeSignature = utils.raw_time_signature(midi_obj, i.start)
        ############################################################
        if len(note_items) == 0:
            return [], None
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items

        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups, task)
        return events, pianohist

    def padding(self, data, max_len, ans):
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            if not ans:
                data.append(self.pad_word)
            else:
                data.append(0)

        return data

    def prepare_data(self, midi_paths, task, max_len):
        all_words, all_ys = [], []

        for i, path in enumerate(tqdm(midi_paths)):
            # if i == 100:
            #     break

            # extract events
            events, histp = self.extract_events(path, task)
            if len(events) == 0:
                continue

            # events to words
            words, ys = [], []
            for note_tuple in events:
                nts = []
                for e in note_tuple:
                    if e.name == "genLabel":
                        continue
                    e_text = f"{e.name} {e.value}"
                    nts.append(self.event2word[e.name][e_text])
                words.append(nts)

            if task == "custom":
                slice_words = []
                for i in range(0, len(words), max_len):
                    slice_words.append(words[i : i + max_len])
                if len(slice_words[-1]) < max_len:
                    slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)

            all_words = all_words + list(slice_words)
            # all_ys = all_ys + list(slice_ys)

        all_words = np.array(all_words).astype(np.int64)
        # all_ys = np.array(all_ys).astype(np.int64)
        return all_words, all_ys
