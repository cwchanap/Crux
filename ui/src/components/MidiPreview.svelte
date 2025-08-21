<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { midiStore } from '../stores/midi';
  import abcjs from 'abcjs';
  import { Midi } from '@tonejs/midi';
  
  let notationContainer: HTMLDivElement;
  let abcString = '';
  let isLoading = true;
  
  $: if ($midiStore.midiData && notationContainer) {
    renderNotation($midiStore.midiData);
  }
  
  function midiToAbc(midi: Midi): string {
    // Convert MIDI to ABC notation format
    let abc = 'X:1\n';
    abc += 'T:Drum Transcription\n';
    abc += 'M:4/4\n';
    abc += 'L:1/16\n';
    abc += 'Q:1/4=120\n';
    abc += 'K:C clef=percussion\n';
    
    // Map drum MIDI notes to ABC percussion notation
    const drumMap: { [key: number]: string } = {
      36: 'C', // Bass Drum
      38: 'E', // Snare
      42: 'G', // Hi-Hat Closed
      46: 'A', // Hi-Hat Open
      49: 'c', // Crash
      51: 'd', // Ride
      45: 'F', // Tom Low
      47: 'F', // Tom Mid
      48: 'G', // Tom High
    };
    
    if (midi.tracks.length > 0) {
      const track = midi.tracks[0];
      const notes = [...track.notes].sort((a, b) => a.time - b.time);
      
      let currentMeasure = '';
      let currentBeat = 0;
      let measuresWritten = 0;
      const beatsPerMeasure = 16; // 16 sixteenth notes per measure
      
      for (const note of notes) {
        const noteStart = Math.round(note.time * 4); // Convert to 16th notes
        const noteDuration = Math.max(1, Math.round(note.duration * 4));
        
        // Fill rests if needed
        while (currentBeat < noteStart % beatsPerMeasure) {
          currentMeasure += 'z';
          currentBeat++;
          
          if (currentBeat >= beatsPerMeasure) {
            abc += currentMeasure + '|';
            measuresWritten++;
            if (measuresWritten % 4 === 0) abc += '\n';
            currentMeasure = '';
            currentBeat = 0;
          }
        }
        
        // Add the note
        const noteName = drumMap[note.midi] || 'C';
        currentMeasure += noteName;
        currentBeat += noteDuration;
        
        // Handle measure overflow
        if (currentBeat >= beatsPerMeasure) {
          abc += currentMeasure + '|';
          measuresWritten++;
          if (measuresWritten % 4 === 0) abc += '\n';
          currentMeasure = '';
          currentBeat = currentBeat % beatsPerMeasure;
        }
      }
      
      // Add final measure if not empty
      if (currentMeasure) {
        while (currentBeat < beatsPerMeasure) {
          currentMeasure += 'z';
          currentBeat++;
        }
        abc += currentMeasure + '|';
      }
    }
    
    return abc;
  }
  
  function renderNotation(midi: Midi) {
    isLoading = true;
    try {
      abcString = midiToAbc(midi);
      
      // Clear previous notation
      notationContainer.innerHTML = '';
      
      // Render ABC notation
      abcjs.renderAbc(notationContainer, abcString, {
        responsive: 'resize',
        staffwidth: 800,
        wrap: {
          minSpacing: 1.5,
          maxSpacing: 2.5,
          preferredMeasuresPerLine: 4
        },
        clickListener: (abcElem: any) => {
          if (abcElem.startTime !== undefined) {
            midiStore.seek(abcElem.startTime);
          }
        }
      });
      
      isLoading = false;
    } catch (error) {
      console.error('Error rendering notation:', error);
      isLoading = false;
    }
  }
  
  function formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
  
  function handleSeek(e: Event) {
    const target = e.target as HTMLInputElement;
    midiStore.seek(parseFloat(target.value));
  }
  
  function handleClose() {
    midiStore.close();
  }
</script>

{#if $midiStore.isOpen}
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
    <div class="bg-white rounded-2xl shadow-2xl max-w-6xl w-full max-h-[90vh] flex flex-col">
      <!-- Header -->
      <div class="p-6 border-b border-gray-200 flex justify-between items-center">
        <h2 class="text-2xl font-bold text-gray-800">
          🎼 MIDI Preview
        </h2>
        <button
          on:click={handleClose}
          class="text-gray-500 hover:text-gray-700 text-2xl"
        >
          ✕
        </button>
      </div>
      
      <!-- Notation Display -->
      <div class="flex-1 overflow-auto p-6 bg-gray-50">
        {#if isLoading}
          <div class="flex justify-center items-center h-64">
            <div class="inline-block w-8 h-8 border-4 border-gray-300 border-t-purple-600 rounded-full animate-spin"></div>
          </div>
        {:else}
          <div bind:this={notationContainer} class="notation-container"></div>
        {/if}
      </div>
      
      <!-- Playback Controls -->
      <div class="p-6 border-t border-gray-200 bg-white">
        <!-- Progress Bar -->
        <div class="mb-4">
          <input
            type="range"
            min="0"
            max={$midiStore.duration}
            value={$midiStore.currentTime}
            on:input={handleSeek}
            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
          />
          <div class="flex justify-between text-sm text-gray-600 mt-1">
            <span>{formatTime($midiStore.currentTime)}</span>
            <span>{formatTime($midiStore.duration)}</span>
          </div>
        </div>
        
        <!-- Control Buttons -->
        <div class="flex justify-center space-x-4">
          <button
            on:click={() => midiStore.stop()}
            class="px-6 py-3 bg-gray-200 hover:bg-gray-300 rounded-full transition-colors"
            title="Stop"
          >
            ⏹️
          </button>
          
          {#if $midiStore.isPlaying}
            <button
              on:click={() => midiStore.pause()}
              class="px-8 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-full transition-colors"
              title="Pause"
            >
              ⏸️ Pause
            </button>
          {:else}
            <button
              on:click={() => midiStore.play()}
              class="px-8 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-full transition-colors"
              title="Play"
            >
              ▶️ Play
            </button>
          {/if}
          
          <a
            href="/api/jobs/{$midiStore.jobId}/download"
            download
            class="px-6 py-3 bg-green-500 hover:bg-green-600 text-white rounded-full transition-colors"
            title="Download"
          >
            ⬇️ Download
          </a>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  :global(.notation-container svg) {
    max-width: 100%;
    height: auto;
  }
  
  :global(.abcjs-cursor) {
    fill: #9333ea;
    opacity: 0.5;
  }
  
  .slider::-webkit-slider-thumb {
    appearance: none;
    width: 16px;
    height: 16px;
    background: #9333ea;
    cursor: pointer;
    border-radius: 50%;
  }
  
  .slider::-moz-range-thumb {
    width: 16px;
    height: 16px;
    background: #9333ea;
    cursor: pointer;
    border-radius: 50%;
    border: none;
  }
</style>
