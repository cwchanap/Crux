<script lang="ts">
  import { jobsStore } from '../stores/jobs';
  import { toastStore } from '../stores/toast';

  let dragover = false;
  let fileInput: HTMLInputElement;
  
  const validTypes = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/mp4', 'audio/x-m4a', 'audio/flac'];
  
  async function handleFileUpload(file: File) {
    // Validate file size
    if (file.size > 50 * 1024 * 1024) {
      toastStore.show('File too large. Maximum size is 50MB.', 'error');
      return;
    }
    
    // Validate file type
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp3|wav|m4a|flac)$/i)) {
      toastStore.show('Invalid file type. Please upload MP3, WAV, M4A, or FLAC.', 'error');
      return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      toastStore.show('Uploading file...', 'success');
      
      const response = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      toastStore.show(`Job created: ${result.job_id.substring(0, 8)}...`, 'success');
      
      // Start auto-refresh and reload jobs
      jobsStore.startAutoRefresh();
      jobsStore.loadJobs();
      
    } catch (error) {
      console.error('Upload error:', error);
      toastStore.show(`Upload failed: ${error.message}`, 'error');
    }
  }
  
  function handleDrop(e: DragEvent) {
    e.preventDefault();
    dragover = false;
    
    if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  }
  
  function handleFileSelect(e: Event) {
    const target = e.target as HTMLInputElement;
    if (target.files && target.files.length > 0) {
      handleFileUpload(target.files[0]);
    }
  }
</script>

<div 
  class="border-3 border-dashed border-purple-500 rounded-2xl p-10 text-center transition-all duration-300 cursor-pointer bg-purple-50 hover:bg-purple-100"
  class:border-purple-700={dragover}
  class:bg-purple-100={dragover}
  class:scale-105={dragover}
  on:click={() => fileInput.click()}
  on:dragover|preventDefault={() => dragover = true}
  on:dragleave={() => dragover = false}
  on:drop={handleDrop}
  on:keydown={(e) => e.key === 'Enter' && fileInput.click()}
  role="button"
  tabindex="0"
>
  <div class="text-6xl mb-5">📁</div>
  <div class="text-xl text-gray-700 mb-2">
    Drop your audio file here or click to browse
  </div>
  <div class="text-sm text-gray-500">
    Supports MP3, WAV, M4A, FLAC (max 50MB)
  </div>
  
  <input 
    type="file" 
    class="hidden"
    accept=".mp3,.wav,.m4a,.flac"
    bind:this={fileInput}
    on:change={handleFileSelect}
  />
</div>
