<script lang="ts">
  import { toastStore, type Toast } from '../stores/toast';
  import { fly } from 'svelte/transition';
  
  let toasts: Toast[] = [];
  
  toastStore.subscribe(value => {
    toasts = value;
  });
  
  $: toastClasses = (type: string) => {
    const baseClasses = 'fixed bottom-8 right-8 bg-white px-6 py-4 rounded-xl shadow-2xl flex items-center space-x-3 min-w-[300px]';
    const borderClasses = {
      'success': 'border-l-4 border-green-500',
      'error': 'border-l-4 border-red-500',
      'info': 'border-l-4 border-blue-500'
    };
    return `${baseClasses} ${borderClasses[type] || ''}`;
  };
  
  $: iconForType = (type: string) => {
    const icons = {
      'success': '✅',
      'error': '❌',
      'info': 'ℹ️'
    };
    return icons[type] || '✅';
  };
</script>

{#each toasts as toast (toast.id)}
  <div 
    class={toastClasses(toast.type)}
    transition:fly={{ x: 100, duration: 300 }}
  >
    <span class="text-2xl">{iconForType(toast.type)}</span>
    <span class="text-gray-700">{toast.message}</span>
  </div>
{/each}
