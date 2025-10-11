const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const glob = require('glob');

// Configuration paths
const htmlFolderPath = '../data/last_trajectory_htmls';
const imageFolderPath = '../data/last_trajectory_images';

// Ensure output folder exists
if (!fs.existsSync(imageFolderPath)) {
  fs.mkdirSync(imageFolderPath, { recursive: true });
}

// Configuration
const config = {
  concurrency: 4,  // Concurrency count
  viewportWidth: 1280,
  viewportHeight: 1280,
  waitTime: 12000,  // Wait time (milliseconds)
};

// Get all HTML files
function getHtmlFiles() {
  return glob.sync(path.join(htmlFolderPath, 'trajectory_*.html'));
}

// Create a screenshot task
async function takeScreenshot(htmlPath) {
  const fileName = path.basename(htmlPath);
  const id = fileName.replace('.html', '');
  const pngPath = path.join(imageFolderPath, fileName.replace('.html', '.png'));

  try {
    console.log(`Processing file: ${fileName}`);

    // Check if output file already exists
    if (fs.existsSync(pngPath)) {
      console.log(`Screenshot already exists: ${pngPath}`);
      return { id, status: 'skipped' };
    }

    // Launch browser
    const browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    try {
      const page = await browser.newPage();
      await page.setViewport({
        width: config.viewportWidth,
        height: config.viewportHeight
      });

      // Convert to file URL
      const fileUrl = 'file://' + path.resolve(htmlPath);
      await page.goto(fileUrl, { waitUntil: 'networkidle0' });

      // Wait for map to fully load - use setTimeout instead of waitForTimeout
      await new Promise(resolve => setTimeout(resolve, config.waitTime));

      // Take screenshot
      await page.screenshot({ path: pngPath, fullPage: true });
      console.log(`Screenshot saved: ${pngPath}`);

      return { id, status: 'success' };
    } finally {
      await browser.close();
    }
  } catch (error) {
    console.error(`Error processing file ${fileName}:`, error);
    return { id, status: 'error', error: error.toString() };
  }
}

// Use Promise.all to handle concurrency
async function processInBatches(items, batchSize) {
  const results = [];

  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    console.log(`Processing batch ${i / batchSize + 1}/${Math.ceil(items.length / batchSize)}`);

    const batchResults = await Promise.all(batch.map(takeScreenshot));
    results.push(...batchResults);
  }

  return results;
}

// Main function
async function main() {
  const htmlFiles = getHtmlFiles();
  console.log(`Found ${htmlFiles.length} HTML files to process`);

  const startTime = Date.now();

  try {
    const results = await processInBatches(htmlFiles, config.concurrency);

    // Calculate statistics
    const successCount = results.filter(r => r.status === 'success').length;
    const errorCount = results.filter(r => r.status === 'error').length;
    const skippedCount = results.filter(r => r.status === 'skipped').length;

    console.log('===== Processing Result Statistics =====');
    console.log(`Total processed: ${results.length} screenshots`);
    console.log(`Successfully processed: ${successCount}`);
    console.log(`Processing errors: ${errorCount}`);
    console.log(`Skipped processing: ${skippedCount}`);
    console.log('=======================================');

    // Save processing results
    fs.writeFileSync(
      path.join(imageFolderPath, 'screenshot_results.json'),
      JSON.stringify(results, null, 2)
    );

  } catch (error) {
    console.error('Error occurred during processing:', error);
  }

  const endTime = Date.now();
  console.log(`Total time: ${((endTime - startTime) / 1000).toFixed(2)} seconds`);
}

main();