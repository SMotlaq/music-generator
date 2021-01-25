import MIDI
import Model

if __name__ == '__main__':
    midi = MIDI.MIDI(seq_length=100)
    model = Model.MODEL(midi_obj=midi)
    model.train(2000, dataFolder='Data2', batch_size=128, sample_interval=1)
    model.save()
    model.generate()
    model.plot_loss()
