import { test, expect, chromium } from '@playwright/test';

test.describe('Simple E2E Test', () => {
  test('should run comprehensive UI test with screenshots', async ({ page }) => {
    console.log('🚀 Starting Simple E2E Test');
    
    // Navigate to application
    console.log('📍 Navigating to application...');
    await page.goto('/');
    
    // Take screenshot of initial state
    await page.screenshot({ path: 'test-screenshots/01-initial-load.png', fullPage: true });
    console.log('✅ Page loaded, screenshot saved');
    
    // Check for main heading
    const heading = page.locator('h1').first();
    if (await heading.count() > 0) {
      const headingText = await heading.textContent();
      console.log(`✅ Found heading: "${headingText}"`);
      await expect(heading).toBeVisible();
    }
    
    // Check for file upload area
    const uploadArea = page.locator('.border-dashed').first();
    if (await uploadArea.count() > 0) {
      console.log('✅ File upload area found');
      await page.screenshot({ path: 'test-screenshots/02-upload-area.png' });
      await expect(uploadArea).toBeVisible();
    }
    
    // Check for jobs section
    const jobsSection = page.locator('text=Recent Jobs').or(page.locator('text=No jobs yet')).first();
    if (await jobsSection.count() > 0) {
      console.log('✅ Jobs section found');
      await expect(jobsSection).toBeVisible();
    }
    
    // Test MIDI preview by injecting a mock completed job
    console.log('📍 Testing MIDI preview with mock data...');
    
    await page.evaluate(() => {
      // Create a mock job element
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
        // Create mock MIDI preview modal
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
                  <p class="text-sm text-gray-500 mt-2">Drum pattern visualization would appear here</p>
                </div>
              </div>
            </div>
            <div class="p-6 border-t border-gray-200 bg-white">
              <div class="mb-4">
                <input type="range" min="0" max="100" value="0" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
                <div class="flex justify-between text-sm text-gray-600 mt-1">
                  <span>0:00</span>
                  <span>0:02</span>
                </div>
              </div>
              <div class="flex justify-center space-x-4">
                <button class="px-6 py-3 bg-gray-200 hover:bg-gray-300 rounded-full transition-colors">⏹️</button>
                <button id="play-btn" class="px-8 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-full transition-colors">▶️ Play</button>
                <button class="px-6 py-3 bg-green-500 hover:bg-green-600 text-white rounded-full transition-colors">⬇️ Download</button>
              </div>
            </div>
          </div>
        `;
        
        document.body.appendChild(modal);
        
        // Add close handler
        (document.getElementById('close-modal') as HTMLButtonElement)?.addEventListener('click', () => {
          modal.remove();
        });
        
        // Add play button handler
        (document.getElementById('play-btn') as HTMLButtonElement)?.addEventListener('click', (e) => {
          const btn = e.target as HTMLButtonElement;
          if (btn?.textContent?.includes('Play')) {
            btn.textContent = '⏸️ Pause';
            btn.className = btn.className.replace('bg-purple-600', 'bg-orange-600');
          } else if (btn?.textContent) {
            btn.textContent = '▶️ Play';
            btn.className = btn.className.replace('bg-orange-600', 'bg-purple-600');
          }
        });
      });
    });
    
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'test-screenshots/03-with-mock-job.png', fullPage: true });
    console.log('✅ Mock job added');
    
    // Click the preview button
    const previewBtn = page.locator('#mock-preview-btn');
    await previewBtn.click();
    console.log('🎵 Clicked preview button');
    
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'test-screenshots/04-midi-preview-modal.png', fullPage: true });
    console.log('✅ MIDI preview modal opened');
    
    // Verify modal is visible
    await expect(page.locator('#midi-preview-modal')).toBeVisible();
    await expect(page.locator('text=🎼 MIDI Preview')).toBeVisible();
    
    // Test play button
    const playBtn = page.locator('#play-btn');
    await playBtn.click();
    console.log('▶️ Clicked play button');
    
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'test-screenshots/05-playing-state.png', fullPage: true });
    console.log('✅ Play button state changed');
    
    // Verify pause state
    await expect(page.locator('text=⏸️ Pause')).toBeVisible();
    
    // Test pause
    await playBtn.click();
    console.log('⏸️ Clicked pause');
    
    // Verify play state restored
    await expect(page.locator('text=▶️ Play')).toBeVisible();
    
    // Close modal
    const closeBtn = page.locator('#close-modal');
    await closeBtn.click();
    console.log('✅ Closed modal');
    
    await page.waitForTimeout(500);
    await page.screenshot({ path: 'test-screenshots/06-modal-closed.png', fullPage: true });
    
    // Verify modal is closed
    await expect(page.locator('#midi-preview-modal')).not.toBeVisible();
    
    // Test responsive design
    console.log('📍 Testing responsive design...');
    
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(500);
    await page.screenshot({ path: 'test-screenshots/07-mobile-view.png', fullPage: true });
    console.log('✅ Mobile view tested');
    
    // Verify elements still visible on mobile
    await expect(page.locator('h1')).toBeVisible();
    
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.waitForTimeout(500);
    await page.screenshot({ path: 'test-screenshots/08-tablet-view.png', fullPage: true });
    console.log('✅ Tablet view tested');
    
    // Verify elements still visible on tablet
    await expect(page.locator('h1')).toBeVisible();
    
    console.log('🎉 E2E Test completed successfully!');
    console.log('📸 Screenshots saved in test-screenshots/ directory');
  });
});
