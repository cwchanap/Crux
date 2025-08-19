<script lang="ts">
  import { jobsStore, type Job } from '../stores/jobs';
  import JobCard from './JobCard.svelte';

  let jobs: Job[] = [];
  
  jobsStore.subscribe(value => {
    jobs = value;
  });
  
  function refresh() {
    jobsStore.loadJobs();
  }
</script>

<div class="jobs-section">
  <div class="flex justify-between items-center mb-6">
    <h2 class="text-2xl font-semibold text-gray-800">Recent Jobs</h2>
    <button 
      on:click={refresh}
      class="bg-purple-50 text-purple-600 border-2 border-purple-600 px-5 py-2 rounded-full hover:bg-purple-600 hover:text-white transition-all duration-300 text-sm font-medium"
    >
      🔄 Refresh
    </button>
  </div>
  
  <div class="space-y-4">
    {#if jobs.length === 0}
      <div class="text-center py-10 text-gray-400">
        No jobs yet. Upload an audio file to get started!
      </div>
    {:else}
      {#each jobs as job (job.job_id)}
        <JobCard {job} />
      {/each}
    {/if}
  </div>
</div>
