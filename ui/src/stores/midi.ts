import { writable } from 'svelte/store';
import { Midi } from '@tonejs/midi';
import * as Tone from 'tone';

export interface MidiPreviewState {
  isOpen: boolean;
  jobId: string | null;
  midiData: Midi | null;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
}

function createMidiStore() {
  const { subscribe, set, update } = writable<MidiPreviewState>({
    isOpen: false,
    jobId: null,
    midiData: null,
    isPlaying: false,
    currentTime: 0,
    duration: 0
  });

  let synth: Tone.PolySynth | null = null;
  let transport: typeof Tone.Transport = Tone.Transport;
  let scheduledEvents: number[] = [];

  const openPreview = async (jobId: string) => {
    try {
      // Fetch MIDI file
      const response = await fetch(`/api/jobs/${jobId}/download`);
      if (!response.ok) throw new Error('Failed to fetch MIDI file');
      
      const arrayBuffer = await response.arrayBuffer();
      const midi = new Midi(arrayBuffer);
      
      update(state => ({
        ...state,
        isOpen: true,
        jobId,
        midiData: midi,
        duration: midi.duration
      }));
      
      // Initialize synth for playback
      if (!synth) {
        synth = new Tone.PolySynth(Tone.Synth, {
          envelope: {
            attack: 0.02,
            decay: 0.1,
            sustain: 0.3,
            release: 0.5
          }
        }).toDestination();
      }
      
      // Schedule MIDI events for playback
      scheduleEvents(midi);
      
    } catch (error) {
      console.error('Error loading MIDI:', error);
    }
  };

  const scheduleEvents = (midi: Midi) => {
    // Clear previous events
    scheduledEvents.forEach(id => transport.clear(id));
    scheduledEvents = [];
    
    // Schedule new events from all tracks
    midi.tracks.forEach(track => {
      track.notes.forEach(note => {
        const eventId = transport.schedule(time => {
          synth?.triggerAttackRelease(
            note.name,
            note.duration,
            time,
            note.velocity
          );
        }, note.time);
        scheduledEvents.push(eventId);
      });
    });
  };

  const play = async () => {
    await Tone.start();
    transport.start();
    update(state => ({ ...state, isPlaying: true }));
  };

  const pause = () => {
    transport.pause();
    update(state => ({ ...state, isPlaying: false }));
  };

  const stop = () => {
    transport.stop();
    transport.position = 0;
    update(state => ({ 
      ...state, 
      isPlaying: false,
      currentTime: 0
    }));
  };

  const seek = (time: number) => {
    transport.position = time;
    update(state => ({ ...state, currentTime: time }));
  };

  const close = () => {
    stop();
    set({
      isOpen: false,
      jobId: null,
      midiData: null,
      isPlaying: false,
      currentTime: 0,
      duration: 0
    });
  };

  // Update current time during playback
  setInterval(() => {
    update(state => {
      if (state.isPlaying) {
        return { ...state, currentTime: transport.position };
      }
      return state;
    });
  }, 100);

  return {
    subscribe,
    openPreview,
    play,
    pause,
    stop,
    seek,
    close
  };
}

export const midiStore = createMidiStore();
