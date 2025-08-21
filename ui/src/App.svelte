<script lang="ts">
  import { onMount } from 'svelte';
  import FileUpload from './components/FileUpload.svelte';
  import JobsList from './components/JobsList.svelte';
  import Toast from './components/Toast.svelte';
  import MidiPreview from './components/MidiPreview.svelte';
  import { jobsStore } from './stores/jobs';
  import { toastStore } from './stores/toast';

  onMount(() => {
    // Load initial jobs
    jobsStore.loadJobs();
  });
</script>

<div class="min-h-screen bg-gradient-to-br from-purple-500 via-purple-600 to-purple-700 flex items-center justify-center p-5">
  <div class="bg-white rounded-2xl shadow-2xl max-w-4xl w-full p-10">
    <header class="text-center mb-10">
      <h1 class="text-4xl font-bold text-gray-800 mb-2">
        🥁 Drum Transcription
      </h1>
      <p class="text-gray-600 text-lg">
        Upload your drum audio and get MIDI transcription powered by AI
      </p>
    </header>

    <FileUpload />
    
    <div class="mt-10 pt-10 border-t border-gray-200">
      <JobsList />
    </div>
  </div>
</div>

<Toast />
<MidiPreview />
