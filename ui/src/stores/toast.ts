import { writable } from 'svelte/store';

export type ToastType = 'success' | 'error' | 'info';

export interface Toast {
  message: string;
  type: ToastType;
  id: number;
}

function createToastStore() {
  const { subscribe, update } = writable<Toast[]>([]);
  let nextId = 0;

  const show = (message: string, type: ToastType = 'success') => {
    const id = nextId++;
    
    update(toasts => [...toasts, { message, type, id }]);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
      update(toasts => toasts.filter(t => t.id !== id));
    }, 3000);
  };

  return {
    subscribe,
    show
  };
}

export const toastStore = createToastStore();
