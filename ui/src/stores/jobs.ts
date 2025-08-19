import { writable } from 'svelte/store';
import { toastStore } from './toast';

export interface Job {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  progress: number;
  result_url?: string;
  error?: string;
  metadata?: {
    filename: string;
    file_size: number;
  };
}

function createJobsStore() {
  const { subscribe, set, update } = writable<Job[]>([]);
  let autoRefreshInterval: number | null = null;

  const loadJobs = async () => {
    try {
      const response = await fetch('/api/jobs?limit=10');
      if (!response.ok) throw new Error('Failed to load jobs');
      
      const data = await response.json();
      set(data.jobs);
      
      // Check if any jobs are still processing
      const hasProcessing = data.jobs.some((job: Job) => 
        job.status === 'pending' || job.status === 'processing'
      );
      
      if (hasProcessing && !autoRefreshInterval) {
        startAutoRefresh();
      } else if (!hasProcessing && autoRefreshInterval) {
        stopAutoRefresh();
      }
    } catch (error) {
      console.error('Error loading jobs:', error);
      toastStore.show('Failed to load jobs', 'error');
    }
  };

  const startAutoRefresh = () => {
    if (autoRefreshInterval) return;
    autoRefreshInterval = window.setInterval(loadJobs, 2000);
  };

  const stopAutoRefresh = () => {
    if (autoRefreshInterval) {
      clearInterval(autoRefreshInterval);
      autoRefreshInterval = null;
    }
  };

  return {
    subscribe,
    loadJobs,
    startAutoRefresh,
    stopAutoRefresh
  };
}

export const jobsStore = createJobsStore();
