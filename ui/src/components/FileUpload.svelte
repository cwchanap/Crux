<script lang="ts">
  import { jobsStore } from '../stores/jobs';
  import { toastStore } from '../stores/toast';

  let dragover = false;
  let fileInput: HTMLInputElement;
  let uploadedFile: { upload_id: string; filename: string; file_size: number } | null = null;
  let isStarting = false;
  
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
      
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      uploadedFile = {
        upload_id: result.upload_id,
        filename: result.filename,
        file_size: result.file_size
      };
      
      toastStore.show('File uploaded successfully! Click Start to begin transcription.', 'success');
      
    } catch (error) {
      console.error('Upload error:', error);
      toastStore.show(`Upload failed: ${error.message}`, 'error');
    }
  }

  async function startTranscription() {
    if (!uploadedFile) return;
    
    isStarting = true;
    
    try {
      const response = await fetch('/api/transcribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          upload_id: uploadedFile.upload_id
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to start transcription: ${response.statusText}`);
      }
      
      const result = await response.json();
      toastStore.show(`Transcription started: ${result.job_id.substring(0, 8)}...`, 'success');
      
      // Start auto-refresh and reload jobs
      jobsStore.startAutoRefresh();
      jobsStore.loadJobs();
      
      // Reset the upload state
      uploadedFile = null;
      
    } catch (error) {
      console.error('Start transcription error:', error);
      toastStore.show(`Failed to start transcription: ${error.message}`, 'error');
    } finally {
      isStarting = false;
    }
  }

  function resetUpload() {
    uploadedFile = null;
    if (fileInput) {
      fileInput.value = '';
    }
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
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

{#if !uploadedFile}
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
{:else}
  <div class="bg-green-50 border-2 border-green-200 rounded-2xl p-8">
    <div class="flex items-center justify-between mb-6">
      <div class="flex items-center space-x-4">
        <div class="text-4xl">🎵</div>
        <div>
          <div class="text-lg font-semibold text-gray-800">{uploadedFile.filename}</div>
          <div class="text-sm text-gray-600">{formatFileSize(uploadedFile.file_size)}</div>
        </div>
      </div>
      <button
        on:click={resetUpload}
        class="text-gray-500 hover:text-gray-700 transition-colors"
        title="Remove file"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
        </svg>
      </button>
    </div>
    
    <div class="flex space-x-4">
      <button
        on:click={startTranscription}
        disabled={isStarting}
        class="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 text-white font-semibold py-3 px-6 rounded-xl transition-colors duration-200 flex items-center justify-center space-x-2"
      >
        {#if isStarting}
          <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span>Starting...</span>
        {:else}
          <span>🚀 Start Transcription</span>
        {/if}
      </button>
      
      <button
        on:click={resetUpload}
        class="bg-gray-200 hover:bg-gray-300 text-gray-700 font-semibold py-3 px-6 rounded-xl transition-colors duration-200"
      >
        Upload Different File
      </button>
    </div>
  </div>
{/if}
