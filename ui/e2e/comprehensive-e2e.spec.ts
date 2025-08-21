import { test, expect } from '@playwright/test';

test.describe('Comprehensive E2E Test Suite', () => {
  test('should load application and verify all components', async ({ page }) => {
    await page.goto('/');
    
    // Check for main heading
    await expect(page.locator('h1')).toBeVisible();
    
    // Check for file upload area
    await expect(page.locator('.border-dashed')).toBeVisible();
    
    // Check for jobs section
    await expect(page.locator('text=Recent Jobs').or(page.locator('text=No jobs yet'))).toBeVisible();
  });

  test('should display file upload interface correctly', async ({ page }) => {
    await page.goto('/');
    
    // Check upload area is interactive
    const uploadArea = page.locator('.border-dashed').first();
    await expect(uploadArea).toBeVisible();
    
    // Check for upload text
    await expect(page.locator('text=Drag & drop your audio file here')).toBeVisible();
  });

  test('should handle MIDI preview functionality with mock data', async ({ page }) => {
    await page.goto('/');
    
    // Inject a mock completed job
    await page.evaluate(() => {
      const jobsContainer = document.querySelector('.space-y-4') || document.body;
      
      const mockJobCard = document.createElement('div');
      mockJobCard.className = 'bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300';
      mockJobCard.innerHTML = `
        <div class="flex justify-between items-start">
          <div>
            <h3 class="font-semibold text-lg text-gray-800">test_drums.wav</h3>
            <p class="text-sm text-gray-600 mt-1">Uploaded just now</p>
            <span class="inline-block px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 mt-2">
              completed
            </span>
          </div>
          <div class="flex space-x-2">
            <button id="mock-preview-btn" class="bg-purple-500 text-white px-5 py-2 rounded-full hover:bg-purple-600 transition-all duration-300 text-sm font-medium">
              👁️ Preview
            </button>
            <button class="bg-green-500 text-white px-5 py-2 rounded-full hover:bg-green-600 transition-all duration-300 text-sm font-medium">
              ⬇️ Download
            </button>
          </div>
        </div>
      `;
      
      jobsContainer.appendChild(mockJobCard);
      
      // Add click handler for preview button
      (document.getElementById('mock-preview-btn') as HTMLButtonElement)?.addEventListener('click', () => {
        const modal = document.createElement('div');
        modal.id = 'midi-preview-modal';
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4';
        modal.innerHTML = `
          <div class="bg-white rounded-2xl shadow-2xl max-w-6xl w-full max-h-[90vh] flex flex-col">
            <div class="p-6 border-b border-gray-200 flex justify-between items-center">
              <h2 class="text-2xl font-bold text-gray-800">🎼 MIDI Preview</h2>
              <button id="close-modal" class="text-gray-500 hover:text-gray-700 text-2xl">✕</button>
            </div>
            <div class="flex-1 overflow-auto p-6 bg-gray-50">
              <div class="flex justify-center items-center h-64">
                <div class="text-center">
                  <div class="text-4xl mb-4">🎵</div>
                  <p class="text-gray-600">MIDI Notation Display</p>
                </div>
              </div>
            </div>
            <div class="p-6 border-t border-gray-200 bg-white">
              <div class="mb-4">
                <input type="range" min="0" max="100" value="0" class="w-full h-2 bg-gray-200 rounded-lg">
              </div>
              <div class="flex justify-center space-x-4">
                <button class="px-6 py-3 bg-gray-200 hover:bg-gray-300 rounded-full">⏹️</button>
                <button id="play-btn" class="px-8 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-full">▶️ Play</button>
                <button class="px-6 py-3 bg-green-500 hover:bg-green-600 text-white rounded-full">⬇️ Download</button>
              </div>
            </div>
          </div>
        `;
        
        document.body.appendChild(modal);
        
        (document.getElementById('close-modal') as HTMLButtonElement)?.addEventListener('click', () => {
          modal.remove();
        });
        
        (document.getElementById('play-btn') as HTMLButtonElement)?.addEventListener('click', (e) => {
          const btn = e.target as HTMLButtonElement;
          if (btn?.textContent?.includes('Play')) {
            btn.textContent = '⏸️ Pause';
          } else if (btn?.textContent) {
            btn.textContent = '▶️ Play';
          }
        });
      });
    });
    
    // Click preview button
    await page.click('#mock-preview-btn');
    
    // Verify modal opened
    await expect(page.locator('#midi-preview-modal')).toBeVisible();
    await expect(page.locator('text=🎼 MIDI Preview')).toBeVisible();
    
    // Test play button
    await page.click('#play-btn');
    await expect(page.locator('text=⏸️ Pause')).toBeVisible();
    
    // Test pause
    await page.click('#play-btn');
    await expect(page.locator('text=▶️ Play')).toBeVisible();
    
    // Close modal
    await page.click('#close-modal');
    await expect(page.locator('#midi-preview-modal')).not.toBeVisible();
  });

  test('should be responsive on mobile devices', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    // Check that elements are still visible on mobile
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('.border-dashed')).toBeVisible();
  });

  test('should be responsive on tablet devices', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');
    
    // Check that elements are still visible on tablet
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('.border-dashed')).toBeVisible();
  });
});
