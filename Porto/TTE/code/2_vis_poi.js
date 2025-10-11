const fs = require('fs');
const { execSync } = require('child_process');
const path = require('path');
const puppeteer = require('puppeteer');

// Ensure screenshot folder exists
const imageDir = '../data/poi_trajectory_images';
if (!fs.existsSync(imageDir)) {
    fs.mkdirSync(imageDir);
}

const htmlDir = '../data/poi_trajectory_htmls';
if (!fs.existsSync(htmlDir)) {
    fs.mkdirSync(htmlDir);
}

// Add data cache
let nodesGeoJsonCache = null;
let edgesGeoJsonCache = null;

// Modified to use OpenStreetMap HTML template
const createHtmlTemplate = (trajectory) => {
    // Ensure o_geo is a coordinate array
    const lineArr = trajectory.o_geo || [];
    if (lineArr.length < 2) {
        console.error('Warning: Insufficient trajectory coordinate points');
    }

    // Convert trajectory coordinate points to JSON string
    const trajectoryPoints = JSON.stringify(lineArr);

    return `<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="initial-scale=1.0, user-scalable=no, width=device-width">
    <title>Trajectory Playback - Device ID: ${trajectory.devid || 'Unknown'}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        html, body, #map {
            height: 100%;
            width: 100%;
            margin: 0;
            padding: 0;
        }
        .info {
            position: absolute;
            top: 10px;
            left: 60px;
            background: rgba(255,255,255,0.8);
            padding: 10px;
            border-radius: 5px;
            z-index: 1000;
        }
        .btn-container {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 10px;
            z-index: 1000;
        }
        .btn {
            padding: 8px 15px;
            background: #1890ff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .btn:hover {
            background: #40a9ff;
        }
        .btn:disabled {
            background: #bfbfbf;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Initialize map
        const map = L.map('map').setView([41.15, -8.61], 17); // Default center point
        
        // Modified L.tileLayer section
        const tileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);
        
        // Add tile event listeners
        tileLayer.on('tileloadstart', function(e) {
            console.log('Starting to load tile:', e.url);
        });
        
        tileLayer.on('tileload', function(e) {
            console.log('Tile loaded successfully:', e.tile.src);
        });
        
        tileLayer.on('tileerror', function(e) {
            console.error('Tile loading failed:', e.tile.src);
        });
        
        // Trajectory data
        const trajectoryData = ${trajectoryPoints};
        const times = ${JSON.stringify(trajectory.times || [])};
        
        // Create trajectory line layer
        const fullPathLine = L.polyline([], {
            color: 'Red',
            weight: 6,
            opacity: 0.6
        }).addTo(map);
        
        // Define start point icon - green circle with "Start" character
        const startIcon = L.divIcon({
          html: \`
            <div style="width: 30px; height: 30px; border-radius: 50%; background-color: #4CAF50; color: white; 
                 display: flex; justify-content: center; align-items: center; font-weight: bold; font-size: 14px;">
              S
            </div>
          \`,
          className: 'custom-start-icon',
          iconSize: [30, 30],
          iconAnchor: [15, 15]
        });
        
        // Define end point icon - red circle with "End" character
        const endIcon = L.divIcon({
          html: \`
            <div style="width: 30px; height: 30px; border-radius: 50%; background-color: #F44336; color: white; 
                 display: flex; justify-content: center; align-items: center; font-weight: bold; font-size: 14px;">
              E
            </div>
          \`,
          className: 'custom-end-icon',
          iconSize: [30, 30],
          iconAnchor: [15, 15]
        });

        
        // Initialize map
        function initMap() {
            if (trajectoryData.length === 0) return;
            
            // Convert coordinate point format to Leaflet accepted format [lat, lng]
            const points = trajectoryData.map(point => [point[1], point[0]]);
            
            // Set complete trajectory line
            fullPathLine.setLatLngs(points);
            // Get start and end point coordinates of trajectory
            const startPoint = points[0];
            const endPoint = points[points.length - 1];
                                                       
            // Set map view to fit entire trajectory
            map.fitBounds(fullPathLine.getBounds(), { padding: [50, 50] });
            const trafficLightIcon = L.icon({
                    iconUrl: 'trafficlight.png',  // Please ensure this icon is in the same directory as HTML
                    iconSize: [30, 30],
                    iconAnchor: [10, 10],
                    popupAnchor: [0, -10]
                });
            // Trigger query once after map is loaded
            // Replace Overpass API call section
            // Trigger query once after map is loaded
            map.whenReady(async () => {
                // Remove traffic light related code
                try {
                // Read locally stored traffic light data
                fetch('porto_traffic_lights_raw.json')
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(\`HTTP error! Status code: \${response.status}\`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('Successfully loaded traffic light data:', data);
                        // Filter traffic lights within current map bounds
                        const bounds = map.getBounds();
                        const visibleLights = data.elements.filter(node => {
                            return node.lat >= bounds.getSouth() &&
                                node.lat <= bounds.getNorth() &&
                                node.lon >= bounds.getWest() &&
                                node.lon <= bounds.getEast();
                        });
                        console.log("Number of traffic lights in current view:", visibleLights.length);
                        // Only filter when fullPathLine has path points
                        if (fullPathLine && fullPathLine.getLatLngs().length > 0) {
                            // Get path line points
                            const pathPoints = fullPathLine.getLatLngs();
                            // Define a buffer distance (unit: meters) to determine if traffic lights are near the path
                            const bufferDistance = 20; // Can be adjusted according to actual needs
                            // Filter traffic lights near the path
                            data.elements.forEach(light => {
                                if (light.lat && light.lon) {
                                    // Check if traffic light is near the path
                                    const isNearPath = pathPoints.some(point => {
                                        // Calculate distance from traffic light to path point
                                        const lightLatLng = L.latLng(light.lat, light.lon);
                                        const distance = lightLatLng.distanceTo(point);
                                        return distance <= bufferDistance;
                                    });
                                    // Only add traffic lights near the path
                                    if (isNearPath) {
                                        L.marker([light.lat, light.lon], {icon: trafficLightIcon})
                                            .addTo(map)
                                            .bindPopup("Traffic Light");
                                    }
                                }
                            });
                        } else {
                            console.log("Trajectory path not drawn, cannot filter traffic lights");
                        }
                    })
                    .catch(error => {
                        console.error('Failed to get traffic light data:', error);
                    });
                console.log("Number of traffic light nodes:", data.elements.length);
    } catch (error) {
        console.error('Failed to get traffic light data:', error);
    }
                // Add start and end point markers
                L.marker(startPoint, {icon: startIcon}).addTo(map);
                L.marker(endPoint, {icon: endIcon}).addTo(map);
            });
        }
        
        // Initialize map
        initMap();
    </script>
</body>
</html>`;
};

// Use Puppeteer to render and capture screenshot
async function renderAndCaptureTrajectory(trajectoryFilePath) {
    let browser = null;
    try {
        const trajectoryData = JSON.parse(fs.readFileSync(trajectoryFilePath, 'utf8'));
        const deviceId = path.basename(trajectoryFilePath, '.json');

        // Check if image already exists
        const imagePath = path.join(imageDir, `${deviceId}.png`);
        if (fs.existsSync(imagePath)) {
            return {
                htmlPath: path.join(htmlDir, `${deviceId}.html`),
                imagePath: imagePath,
                skipped: true
            };
        }

        // Create HTML file
        const htmlContent = createHtmlTemplate(trajectoryData);
        const htmlFilePath = path.join(htmlDir, `${deviceId}.html`);
        fs.writeFileSync(htmlFilePath, htmlContent);

        // Use Puppeteer to take screenshot
        browser = await puppeteer.launch({
            // headless: false,
            args: [
                '--allow-file-access-from-files',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process'
            ]
        });
        const page = await browser.newPage();
        await page.setViewport({ width: 1920, height: 1920 });

        // Use local server URL instead of file system path
        await page.goto(`file://${path.resolve(htmlFilePath)}`, { waitUntil: 'networkidle0' });
        await new Promise(resolve => setTimeout(resolve, 16000));

        // After loading trajectory data on page
        // 1. Get trajectory bounds
        await page.evaluate(() => {
            // Assuming your map instance is 'map', trajectory layer is 'trajectoryLayer'
            const bounds = fullPathLine.getBounds();
            // Calculate expansion in all directions
            const northWest = map.latLngToContainerPoint(bounds.getNorthWest());
            const southEast = map.latLngToContainerPoint(bounds.getSouthEast());
            // Add padding
            const padding = 200;
            // Calculate actual screenshot area needed
            window.trajectoryRect = {
                x: Math.max(0, northWest.x - padding),
                y: Math.max(0, northWest.y - padding),
                width: Math.min(2000, southEast.x - northWest.x + 2 * padding),
                height: Math.min(2000, southEast.y - northWest.y + 2 * padding)
            };
        });

        // 2. Get trajectory element position information
        const clipRect = await page.evaluate(() => window.trajectoryRect);

        // 3. Use clip option to take screenshot, only capture trajectory area
        const screenshotPath = path.join(imageDir, `${deviceId}.png`);
        await page.screenshot({
            path: screenshotPath,
            clip: {
                x: clipRect.x,
                y: clipRect.y,
                width: clipRect.width,
                height: clipRect.height
            }
        });

        // Take screenshot
        await browser.close();

        return {
            htmlPath: htmlFilePath,
            imagePath: screenshotPath
        };
    } catch (error) {
        console.error('Error rendering trajectory:', error);
        if (browser) {
            await browser.close();
        }
        return null;
    }
    finally {
        // Ensure browser is closed whether successful or failed
        if (browser !== null) {
            try {
                await browser.close();
            } catch (closeError) {
                console.error('Error closing browser:', closeError);
            }
        }
    }
}

// Main function
const cliProgress = require('cli-progress');

async function main() {
    // Extract trajectory data
    const trajectoryDir = "../data/poi_trajectory_jsons";

    // Get trajectory JSON file list (async version is better, but has minimal impact on overall loop performance)
    let trajectoryFiles;
    try {
        trajectoryFiles = fs.readdirSync(trajectoryDir)
            .filter(file => file.endsWith('.json'));
    } catch (err) {
        console.error(`Failed to read directory ${trajectoryDir}:`, err);
        return;
    }

    if (!trajectoryFiles || trajectoryFiles.length === 0) {
        console.log('No trajectory files found.');
        return;
    }

    console.log(`Found ${trajectoryFiles.length} trajectory files`);

    // Create a progress bar instance
    const progressBar = new cliProgress.SingleBar({
        format: 'Processing trajectories [{bar}] {percentage}% | {value}/{total} files | Completed: {filename}', // Change filename meaning
        barCompleteChar: '\u2588',
        barIncompleteChar: '\u2591',
        hideCursor: true
    }, cliProgress.Presets.shades_classic);

    // Get number of files to process
    const totalFiles = trajectoryFiles.length;

    // Start progress bar
    progressBar.start(totalFiles, 0, { filename: 'N/A' });

    // --- Parallel processing ---
    const MAX_CONCURRENT_TASKS = 1000; // Set maximum concurrent tasks, adjust based on your system resources
    let currentIndex = 0;
    const results = []; // Store results of all promises (if needed)

    // Define function to execute single task
    const processFile = async (fileIndex) => {
        if (fileIndex >= totalFiles) {
            return; // All files have been assigned
        }

        const file = trajectoryFiles[fileIndex];
        const filePath = path.join(trajectoryDir, file);

        try {
            await renderAndCaptureTrajectory(filePath); // Assuming this is your async function
            results.push({ file, status: 'fulfilled' });
        } catch (error) {
            console.error(`\nFailed to process file ${file}:`, error);
            results.push({ file, status: 'rejected', reason: error });
        } finally {
            // Update progress bar, show recently completed filename
            progressBar.increment(1, { filename: file });
            // Start next task (if any)
            if (currentIndex < totalFiles) {
                await processFile(currentIndex++);
            }
        }
    };

    // Start initial batch of concurrent tasks
    const initialTasks = [];
    for (let i = 0; i < Math.min(MAX_CONCURRENT_TASKS, totalFiles); i++) {
        initialTasks.push(processFile(currentIndex++));
    }

    // Wait for all tasks to complete
    await Promise.all(initialTasks);

    // --- End parallel processing ---

    // Stop progress bar after completion
    progressBar.stop();

    const successfulTasks = results.filter(r => r.status === 'fulfilled').length;
    const failedTasks = totalFiles - successfulTasks;

    console.log(`\nAll trajectory processing completed!`);
    console.log(`Success: ${successfulTasks}, Failed: ${failedTasks}`);
    console.log('Trajectory visualization completed');
}

// Run main function
main().catch(console.error);