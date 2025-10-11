const fs = require('fs');
const { execSync } = require('child_process');
const path = require('path');
const puppeteer = require('puppeteer');

const imageDir = '../data/road_structure_anomaly_images';
if (!fs.existsSync(imageDir)) {
    fs.mkdirSync(imageDir);
}

const htmlDir = '../data/road_structure_anomaly_htmls';
if (!fs.existsSync(htmlDir)) {
    fs.mkdirSync(htmlDir);
}

// Add data cache
let nodesGeoJsonCache = null;
let edgesGeoJsonCache = null;

// Calculate extended bounding box
function calculateExtendedBounds(trajectoryData, buffer) {
    let minLat = Infinity, maxLat = -Infinity;
    let minLng = Infinity, maxLng = -Infinity;

    // Traverse all points in trajectory
    for (const point of trajectoryData) {
        const lng = point[0];
        const lat = point[1];

        minLng = Math.min(minLng, lng);
        maxLng = Math.max(maxLng, lng);
        minLat = Math.min(minLat, lat);
        maxLat = Math.max(maxLat, lat);
    }

    // Extend bounding box
    return {
        minLng: minLng - buffer,
        maxLng: maxLng + buffer,
        minLat: minLat - buffer,
        maxLat: maxLat + buffer
    };
}

// Check if point is within bounds
function isPointInBounds(point, bounds) {
    const lng = point[0];
    const lat = point[1];
    return lng >= bounds.minLng && lng <= bounds.maxLng &&
           lat >= bounds.minLat && lat <= bounds.maxLat;
}

// Check if line segment intersects with bounding box
function isLineInBounds(line, bounds) {
    // Simple check: if any point is within bounding box, then line segment intersects with bounding box
    for (const point of line) {
        if (isPointInBounds(point, bounds)) {
            return true;
        }
    }
    return false;
}

// Calculate minimum distance from point to trajectory
function minDistanceToTrajectory(point, trajectoryData) {
    let minDist = Infinity;

    // Calculate minimum distance from point to trajectory line segments
    for (let i = 0; i < trajectoryData.length - 1; i++) {
        const segStart = trajectoryData[i];
        const segEnd = trajectoryData[i + 1];
        const dist = distanceToSegment(
            [point[1], point[0]],
            [segStart[1], segStart[0]],
            [segEnd[1], segEnd[0]]
        );
        minDist = Math.min(minDist, dist);
    }
    return minDist;
}

// Calculate minimum distance from trajectory points to line segment
function minDistanceFromTrajectoryToSegment(trajectoryData, segment) {
    let minDist = Infinity;

    // Calculate minimum distance from each point on trajectory to line segment
    for (const point of trajectoryData) {
        const dist = distanceToSegment(
            [point[1], point[0]],
            [segment[0][1], segment[0][0]],
            [segment[1][1], segment[1][0]]
        );
        minDist = Math.min(minDist, dist);
    }
    return minDist;
}

// Calculate distance from point to line segment
function distanceToSegment(p, v, w) {
    // Square length of line segment v-w
    const l2 = Math.pow(v[0] - w[0], 2) + Math.pow(v[1] - w[1], 2);
    // If line segment is actually a point, return point-to-point distance
    if (l2 === 0) return distance(p, v);
    // Consider line segment v-w as parameterized line segment: v + t (w - v)
    // Parameter t of projection point = ((p-v) . (w-v)) / |w-v|^2
    const t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2;
    if (t < 0) return distance(p, v);      // Beyond v endpoint
    if (t > 1) return distance(p, w);      // Beyond w endpoint
    // Projection falls on line segment, calculate projection point
    const projection = [
        v[0] + t * (w[0] - v[0]),
        v[1] + t * (w[1] - v[1])
    ];
    return distance(p, projection);
}

// Calculate distance between two points
function distance(p1, p2) {
    const R = 6371000; // Earth radius in meters
    const dLat = (p2[0] - p1[0]) * Math.PI / 180;
    const dLng = (p2[1] - p1[1]) * Math.PI / 180;
    const a =
        Math.sin(dLat/2) * Math.sin(dLat/2) +
        Math.cos(p1[0] * Math.PI / 180) * Math.cos(p2[0] * Math.PI / 180) *
        Math.sin(dLng/2) * Math.sin(dLng/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
}

// Pre-filter nodes on server side
function preFilterNodes(trajectoryData, bounds) {
    // First use bounding box for coarse filtering
    let preFilteredNodes = nodesGeoJsonCache.features.filter(feature => {
        const coords = feature.geometry.coordinates;
        return isPointInBounds(coords, bounds);
    });

    // Distance threshold
    const distanceThreshold = 0.001; // 0.001 degrees, approximately 111 meters
    // Then perform precise distance filtering
    const filteredNodes = {
        type: "FeatureCollection",
        features: preFilteredNodes.filter(feature => {
            const coords = feature.geometry.coordinates;
            const dist = minDistanceToTrajectory(coords, trajectoryData);
            return dist < distanceThreshold * 111000; // Convert to meters
        })
    };
    return filteredNodes;
}

// Pre-filter edges on server side
function preFilterEdges(trajectoryData, bounds) {
    // First use bounding box for coarse filtering
    let preFilteredEdges = edgesGeoJsonCache.features.filter(feature => {
        const coords = feature.geometry.coordinates;
        return isLineInBounds(coords, bounds);
    });

    // Distance threshold
    const distanceThreshold = 0.001; // 0.001 degrees, approximately 111 meters
    // Then perform precise distance filtering
    const filteredEdges = {
        type: "FeatureCollection",
        features: preFilteredEdges.filter(feature => {
            const coords = feature.geometry.coordinates;
            // For each line segment, calculate minimum distance from trajectory points to line segment
            for (let i = 0; i < coords.length - 1; i++) {
                const segment = [coords[i], coords[i+1]];
                const dist = minDistanceFromTrajectoryToSegment(trajectoryData, segment);
                if (dist < distanceThreshold * 111000) { // Convert to meters
                    return true; // Keep this edge if any segment meets the condition
                }
            }
            return false;
        })
    };
    return filteredEdges;
}

// Modified to use OpenStreetMap HTML template
const createHtmlTemplate = (trajectory) => {
    // Ensure o_geo is a coordinate array
    const lineArr = trajectory.o_geo || [];
    if (lineArr.length < 2) {
        console.error('Warning: Insufficient trajectory coordinate points');
    }

    // Convert trajectory coordinate points to JSON string
    const trajectoryPoints = JSON.stringify(lineArr);

    // Calculate trajectory bounding box
    const trajectoryBounds = calculateExtendedBounds(lineArr, 0.01); // Use 0.01 as bounding box extension range

    // Pre-filter data on server side
    const filteredNodes = preFilterNodes(lineArr, trajectoryBounds);
    const filteredEdges = preFilterEdges(lineArr, trajectoryBounds);

    // Convert filtered data to JSON strings
    const nodesDataStr = JSON.stringify(filteredNodes);
    const edgesDataStr = JSON.stringify(filteredEdges);

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
<script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <!-- Add arrow decorator plugin -->
    <script src="https://unpkg.com/leaflet-polylinedecorator/dist/leaflet.polylineDecorator.js"></script>
    <script>
        // Pre-loaded GeoJSON data
        const nodesGeoJson = ${nodesDataStr};
        const edgesGeoJson = ${edgesDataStr};
        
        // Initialize map
        const map = L.map('map').setView([41.15, -8.61], 17); // Default center point
        
        // Modified L.tileLayer section
        const tileLayer = L.tileLayer('http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png', {
          attribution: 'Positron'
        }).addTo(map);
        
        // Trajectory data
        const trajectoryData = ${trajectoryPoints};
        const times = ${JSON.stringify(trajectory.times || [])};
        
        // Create trajectory line layer
        const fullPathLine = L.polyline([], {
            color: 'Lime',
            weight: 10,
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
          iconSize: [10, 10],
          iconAnchor: [5, 5]
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
          iconSize: [10, 10],
          iconAnchor: [5, 5]
        });

        
        // Initialize map
        
        // Convert coordinate point format to Leaflet accepted format [lat, lng]
        const points = trajectoryData.map(point => [point[1], point[0]]);
        
        // Set complete trajectory line
        fullPathLine.setLatLngs(points);
        const startPoint = points[0];
        const endPoint = points[points.length - 1];
        
        // Calculate trajectory bounding box (with buffer)
        const trajectoryBounds = calculateExtendedBounds(trajectoryData, 0.01);
        
        // Directly display pre-filtered nodes and edges (no need to filter again)
        displayFilteredNodes();
        displayFilteredEdges();
        
        // Calculate extended bounding box
        function calculateExtendedBounds(trajectoryData, buffer) {
            let minLat = Infinity, maxLat = -Infinity;
            let minLng = Infinity, maxLng = -Infinity;
            
            // Traverse all points in trajectory
            for (const point of trajectoryData) {
                const lng = point[0];
                const lat = point[1];
                
                minLng = Math.min(minLng, lng);
                maxLng = Math.max(maxLng, lng);
                minLat = Math.min(minLat, lat);
                maxLat = Math.max(maxLat, lat);
            }
            
            // Extend bounding box
            return {
                minLng: minLng - buffer,
                maxLng: maxLng + buffer,
                minLat: minLat - buffer,
                maxLat: maxLat + buffer
            };
        }
        
        // Display pre-filtered nodes
        function displayFilteredNodes() {
            const rangeInfo = calculateTrajectoryRange(trajectoryData);
            
            L.geoJSON(nodesGeoJson, {
                pointToLayer: function (feature, latlng) {
                    return L.circleMarker(latlng, {
                        radius: rangeInfo.pointStyle.radius,
                        fillColor: 'red',
                        color: 'darkred',
                        weight: 1,
                        opacity: rangeInfo.pointStyle.opacity,
                        fillOpacity: 0.6
                    });
                },
                onEachFeature: function (feature, layer) {
                    if (feature.properties) {
                        layer.bindPopup('Node ID: ' + feature.properties.id);
                    }
                }
            }).addTo(map);
        }
        
        // Display pre-filtered edges
        function displayFilteredEdges() {
            const rangeInfo = calculateTrajectoryRange(trajectoryData);
            
            L.geoJSON(edgesGeoJson, {
                style: function (feature) {
                    return {
                        color: 'blue',
                        weight: rangeInfo.pointStyle.width,
                        opacity: rangeInfo.pointStyle.opacity
                    };
                },
                onEachFeature: function (feature, layer) {
                    if (feature.properties) {
                        layer.bindPopup('Edge ID: ' + feature.properties.fid);
                    }
                }
            }).addTo(map);
        }
        
        // Calculate geographic distance between two points (unit: meters)
        function calculateDistance(lat1, lng1, lat2, lng2) {
          const R = 6371000; // Earth radius in meters
          const dLat = (lat2 - lat1) * Math.PI / 180;
          const dLng = (lng2 - lng1) * Math.PI / 180;
          const a =
            Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLng/2) * Math.sin(dLng/2);
          const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
          const distance = R * c;
          return distance;
        }
        
        // Calculate trajectory bounding box and range
        function calculateTrajectoryRange(trajectoryData) {
          let minLat = Infinity, maxLat = -Infinity;
          let minLng = Infinity, maxLng = -Infinity;
          
          // Traverse all points in trajectory
          for (const point of trajectoryData) {
            const lng = point[0];
            const lat = point[1];
            
            minLng = Math.min(minLng, lng);
            maxLng = Math.max(maxLng, lng);
            minLat = Math.min(minLat, lat);
            maxLat = Math.max(maxLat, lat);
          }
          
          // Calculate diagonal distance (can be used as approximate trajectory range)
          const diagonalDistance = calculateDistance(minLat, minLng, maxLat, maxLng);
          
          // Determine point radius and opacity
          let radius, opacity;
          
         
          if (diagonalDistance <= 500) {
            radius = 6;
            opacity = 0.8;
            width = 2.5;
          } else if (diagonalDistance <= 1000) {
            radius = 5.8;
            opacity = 0.8;
            width = 2;
          } else if (diagonalDistance <= 1500) {
            radius = 5;
            opacity = 0.7;
            width = 1.8;
          } else if (diagonalDistance <= 2000) {
            radius = 4.5;
            opacity = 0.7;
            width = 1.5;
          } else {
            radius = 4;
            opacity = 0.6;
            width = 1.2;
          }
          return {
            bounds: {
              minLng: minLng,
              maxLng: maxLng,
              minLat: minLat,
              maxLat: maxLat
            },
            range: diagonalDistance,
            pointStyle: {
              radius: radius,
              fillOpacity: opacity,
              opacity: opacity,
              width: width
            }
          };
        }
 
        // Set map view to fit entire trajectory
        map.fitBounds(fullPathLine.getBounds(), { padding: [50, 50] });
        
    </script>
</body>
</html>`;
};

// Load GeoJSON data at script start
async function loadGeoJsonData() {
    console.log('Loading GeoJSON data...');
    try {
        nodesGeoJsonCache = JSON.parse(fs.readFileSync('../data/road_structure_anomaly_htmls/chengdu_nodes.geojson', 'utf8'));
        console.log(`Loaded node data: ${nodesGeoJsonCache.features.length} nodes`);
        edgesGeoJsonCache = JSON.parse(fs.readFileSync('../data/road_structure_anomaly_htmls/chengdu_edges.geojson', 'utf8'));
        console.log(`Loaded edge data: ${edgesGeoJsonCache.features.length} edges`);
        return true;
    } catch (error) {
        console.error('Failed to load GeoJSON data:', error);
        return false;
    }
}

// Use Puppeteer to render and capture screenshot
async function renderAndCaptureTrajectory(trajectoryFilePath) {
    let browser = null;
    try {
        const trajectoryData = JSON.parse(fs.readFileSync(trajectoryFilePath, 'utf8'));
        const deviceId = path.basename(trajectoryFilePath, '.json');
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
        await new Promise(resolve => setTimeout(resolve, 20));

        // After loading trajectory data on page
        // 1. Get trajectory bounds
        await page.evaluate(() => {
            // Assuming your map instance is 'map', trajectory layer is 'trajectoryLayer'
            const bounds = fullPathLine.getBounds();
            // Calculate expansion in all directions
            const northWest = map.latLngToContainerPoint(bounds.getNorthWest());
            const southEast = map.latLngToContainerPoint(bounds.getSouthEast());
            // Add padding
            const padding = 50;
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
        browser = null; // Clear browser variable to prevent duplicate closing in finally

        return {
            htmlPath: htmlFilePath,
            imagePath: screenshotPath
        };
    }
    catch (error) {
        console.error('Error rendering trajectory:', error);
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

const cliProgress = require('cli-progress');

// Main function
async function main() {
    // Pre-load GeoJSON data
    if (!await loadGeoJsonData()) {
        console.error('GeoJSON data loading failed, cannot continue execution');
        return;
    }

    // Extract trajectory data
    const trajectoryDir = "../data/ano_trajectory_jsons";

    // Get trajectory JSON file list
    let trajectoryFiles;
    try {
        trajectoryFiles = fs.readdirSync(trajectoryDir)
            .filter(file => file.endsWith('.json'));
    } catch (err) {
        console.error(`Error: Failed to read directory ${trajectoryDir}. Please check if path is correct and program has read permissions.`, err.message);
        return; // Exit early if unable to read directory
    }

    if (!trajectoryFiles || trajectoryFiles.length === 0) {
        console.log(`No .json trajectory files found in directory ${trajectoryDir}.`);
        return;
    }

    console.log(`Found ${trajectoryFiles.length} trajectory files`);

    // Create a progress bar instance
    const progressBar = new cliProgress.SingleBar({
        format: 'Processing trajectories [{bar}] {percentage}% | {value}/{total} files | Completed: {filename}',
        barCompleteChar: '\u2588',
        barIncompleteChar: '\u2591',
        hideCursor: true
    }, cliProgress.Presets.shades_classic);

    const totalFiles = trajectoryFiles.length;
    progressBar.start(totalFiles, 0, { filename: 'Preparing...' });

    const MAX_CONCURRENT_TASKS = 1000; // Set maximum concurrent tasks
    let currentIndex = 0;
    let processedCount = 0;
    let successfulTasks = 0;
    let failedTasks = 0;

    // Define function to execute single task
    // When this function completes processing one file, it will try to process the next available file
    const worker = async () => {
        while (currentIndex < totalFiles) {
            const fileIndexToProcess = currentIndex;
            currentIndex++; // Immediately increase index, reserve for next worker or next iteration

            const file = trajectoryFiles[fileIndexToProcess];
            const filePath = path.join(trajectoryDir, file);

            try {
                await renderAndCaptureTrajectory(filePath);
                successfulTasks++;
            } catch (error) {
                // Print error outside progress bar to avoid overwriting progress bar
                // Pause progress bar to clearly display error, then resume
                progressBar.stop();
                console.error(`\nError occurred while processing file ${file}: ${error.message || error}`);
                progressBar.start(totalFiles, processedCount, {filename: file}); // Try to restore progress bar state
                failedTasks++;
            } finally {
                processedCount++;
                progressBar.update(processedCount, { filename: file });
            }
        }
    };

    // Start concurrent "worker" tasks
    const workerPromises = [];
    for (let i = 0; i < Math.min(MAX_CONCURRENT_TASKS, totalFiles); i++) {
        workerPromises.push(worker());
    }

    // Wait for all "workers" to complete all tasks in their queue
    await Promise.all(workerPromises);

    progressBar.stop();
    console.log(`\nAll trajectory processing completed!`); // Add a newline
    console.log(`Total: ${totalFiles} files | Success: ${successfulTasks} | Failed: ${failedTasks}`);
    console.log('Trajectory visualization completed');
}

// Run main function
main().catch(console.error);