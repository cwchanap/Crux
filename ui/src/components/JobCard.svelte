<script lang="ts">
  import type { Job } from '../stores/jobs';
  import { midiStore } from '../stores/midi';
  
  export let job: Job;
  
  function handlePreview() {
    midiStore.openPreview(job.job_id);
  }
  
  $: statusClass = {
    'pending': 'bg-yellow-100 text-yellow-800',
    'processing': 'bg-blue-100 text-blue-800',
    'completed': 'bg-green-100 text-green-800',
    'failed': 'bg-red-100 text-red-800'
  }[job.status];
  
  $: progressPercent = job.progress || 0;
</script>

<div class="bg-purple-50 rounded-xl p-5 flex justify-between items-center hover:translate-x-1 hover:shadow-lg transition-all duration-300">
  <div class="flex-1">
    <div class="font-mono text-gray-500 text-sm mb-1">
      ID: {job.job_id.substring(0, 8)}...
    </div>
    <div class="text-gray-800 font-medium mb-2">
      {job.metadata?.filename || 'Unknown file'}
    </div>
    <span class="inline-block px-3 py-1 rounded-full text-xs font-semibold {statusClass}">
      {job.status}
    </span>
    
    {#if job.status === 'processing'}
      <div class="w-48 h-1.5 bg-gray-200 rounded-full overflow-hidden mt-3">
        <div 
          class="h-full bg-gradient-to-r from-purple-500 to-purple-700 transition-all duration-300"
          style="width: {progressPercent}%"
        ></div>
      </div>
    {/if}
  </div>
  
  <div class="flex space-x-2">
    {#if job.status === 'completed'}
      <button
        on:click={handlePreview}
        class="bg-purple-500 text-white px-5 py-2 rounded-full hover:bg-purple-600 hover:-translate-y-0.5 transition-all duration-300 text-sm font-medium"
        title="Preview MIDI"
      >
        👁️ Preview
      </button>
      <a 
        href="/api/jobs/{job.job_id}/download"
        download
        class="bg-green-500 text-white px-5 py-2 rounded-full hover:bg-green-600 hover:-translate-y-0.5 transition-all duration-300 text-sm font-medium inline-block"
      >
        ⬇️ Download
      </a>
    {:else if job.status === 'processing'}
      <div class="inline-block w-5 h-5 border-3 border-gray-300 border-t-purple-600 rounded-full animate-spin"></div>
    {/if}
  </div>
</div>
