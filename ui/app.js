// Global variables
      let fileCRCs = {};
      let selectedFilePath = null;
      let selectedTreeItem = null;
              let viewportZoom = { 1: 1, 2: 1 };
      let viewportPan = { 1: { x: 0, y: 0 }, 2: { x: 0, y: 0 } };
      let isDragging = { 1: false, 2: false };
      let lastMousePos = { 1: { x: 0, y: 0 }, 2: { x: 0, y: 0 } };
      let viewportData = { 1: null, 2: null };
      let s3TreeData = null;
      let currentFrames = { 1: 0, 2: 0 };
      let totalFrames = { 1: 1, 2: 1 };
      
      // Separate data structures for each eye's frame information
      const leftEyeFrameData = {
        totalFrames: 0,
        currentFrame: 0,
        octFrameCount: 0,
        isOctMode: false,
        eyeData: null
      };
      
      const rightEyeFrameData = {
        totalFrames: 0,
        currentFrame: 0,
        octFrameCount: 0,
        isOctMode: false,
        eyeData: null
      };
      
      let s3StatusChecked = false;
      let s3ConfiguredStatus = null;
      let isStackedLayout = false; // Default to side-by-side layout (left and right eye side by side)

      // E2E specific variables
      let isE2EMode = false;
      let currentE2EFile = null;
      let focusedEye = null;

      // Global variables for layout states
      let isDicomStackedLayout = false; // Default to side-by-side for DICOM view

      // Performance monitoring
      let performanceTimers = { 1: null, 2: null };

      // Active downloads tracking for cancellation

      let abortControllers = new Map();

      // Define custom sort order for tree structure
      const SORT_ORDER = [
        "SCR",
        "Day 1",
        "Day 2",
        "Day 3",
        "Day 4",
        "Day 5",
        "Day 6",
        "Day 7",
        "Week 1",
        "Week 2",
        "Week 3",
        "Week 4",
        "Week 5",
        "Week 6",
        "Week 7",
        "Week 8",
        "Week 9",
        "Week 10",
        "Week 11",
        "Week 12",
      ];
      // Debounce function to prevent rapid successive calls
      function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
          const later = () => {
            clearTimeout(timeout);
            func(...args);
          };
          clearTimeout(timeout);
          timeout = setTimeout(later, wait);
        };
      }

      // Enhanced function to load DICOM with CRC-based caching and cancellation support
      const loadIntoViewportWithPath = debounce(async function(viewportNumber, filePath) {
        const img = document.getElementById(`viewportImage${viewportNumber}`);
        const placeholder = document.querySelector(
          `#viewportContent${viewportNumber} .viewport-placeholder`,
        );
        const errorDiv = document.getElementById(`error${viewportNumber}`);
        const frameSliderContainer = document.getElementById(
          `frameSliderContainer${viewportNumber}`,
        );

        // Create abort controller for cancellation
        const abortController = new AbortController();
        const operationId = `load_${viewportNumber}_${Date.now()}`;



        // Validate file type before processing
        const fileType = validateFileType(filePath);
        console.log(`[DEBUG] Validated file type: ${fileType} for ${filePath}`);
        
        // Handle file type transitions
        const isE2EFile = fileType === 'E2E';
        
        if (isE2EMode && !isE2EFile) {
            // Switching from E2E mode to regular DICOM - reset E2E mode
            resetE2EMode();
        } else if (!isE2EMode && isE2EFile) {
            // Switching from regular DICOM to E2E mode - clear both viewports
            clearViewport(1);
            clearViewport(2);
        } else if (!isE2EFile) {
            // Loading regular DICOM file - clear the specific viewport
            clearViewport(viewportNumber);
        }

        // Set abort controller in progress manager
        progressManagers[viewportNumber].setAbortController(abortController);

        // Initialize cacheSource variable at function scope
        let cacheSource = "unknown";

        // Start performance timer
        startPerformanceTimer(viewportNumber, "File Selection â†’ Image Loaded");

        // Extract file info for metadata
        const fileName = filePath.split("/").pop();
        const fileExt = fileName.split(".").pop()?.toUpperCase() || "UNKNOWN";

        // Show enhanced progress with cancel support
        const progressOperationId = showProgress(
          viewportNumber,
          "Checking cache...",
          {
            fileName: fileName,
            fileType: fileType,
            Type: fileType,
            Extension: fileExt,
            Source: "S3",
            Cache: "Checking...",
          },
        );

        // Ensure progress is visible before proceeding
        await new Promise(resolve => setTimeout(resolve, 200));

        // Hide other elements
        if (placeholder) placeholder.style.display = "none";
        img.style.display = "none";
        errorDiv.style.display = "none";
        frameSliderContainer.classList.remove("active");

        try {
          console.log(`Starting DICOM load process for ${filePath}`);

          // Check if operation was cancelled
          if (abortController.signal.aborted) {
            throw new Error("Operation cancelled by user");
          }

          // Step 1: Check if file is already cached (either in memory or disk)
          nextProgressStep(viewportNumber, "Checking cache...", {
            fileName: fileName,
            fileType: fileType,
            Status: "Cache Check",
            Progress: "Step 1/5",
          });

          // Get file metadata for CRC calculation
          const fileMetadata = {
            path: filePath,
            size: s3Browser.selectedItem?.size || 0,
            lastModified: s3Browser.selectedItem?.last_modified || "",
            frame: 0,
          };

          // Try to load from CRC cache first
          try {
            const cachedResult = await loadImageWithCRC(filePath, fileMetadata);

            // Check if cancelled after cache check
            if (abortController.signal.aborted) {
              throw new Error("Operation cancelled by user");
            }

            if (
              cachedResult.source === "cache" ||
              cachedResult.source === "cache_backend_crc"
            ) {
              // Image loaded from cache - display immediately
              nextProgressStep(viewportNumber, "Loaded from cache!", {
                fileName: fileName,
                fileType: fileType,
                Status: "Cache Hit",
                Cache: cachedResult.source.toUpperCase(),
                CRC: cachedResult.cacheKey.substring(0, 8),
                Progress: "Step 5/5",
              });

              img.onload = () => {
                console.log(
                  `Cached image loaded successfully for viewport ${viewportNumber}`,
                );
                img.style.display = "block";
                centerImage(viewportNumber);
                setupImageInteractions(viewportNumber);

                // Set up single frame (cached images are typically single frame)
                totalFrames[viewportNumber] = 1;
                currentFrames[viewportNumber] = 0;
                setupFrameSlider(viewportNumber);

                hideProgress(viewportNumber);
                endPerformanceTimer(
                  viewportNumber,
                  "File Selection â†’ Image Loaded (Cache)",
                );


              };

              img.src = cachedResult.imageData;
              return;
            }
          } catch (cacheError) {
            if (abortController.signal.aborted) {
              throw new Error("Operation cancelled by user");
            }
            console.warn(`Cache check failed: ${cacheError.message}`);
          }

          // Step 2: Check if file needs to be cached first
          let crc = fileCRCs[filePath];
          if (!crc) {
            nextProgressStep(viewportNumber, "Caching file...", {
              fileName: fileName,
              fileType: fileType,
              Status: "Caching",
              Progress: "Step 2/5",
            });

            try {
              crc = await ensureFileCached(filePath);
            } catch (err) {
              showNotification(err.message, "error");
              hideProgress(viewportNumber);
              return;
            }
          }

          // Step 3: Download and process DICOM from S3 (should use cache if available)
          nextProgressStep(viewportNumber, "Preparing download...", {
            fileName: fileName,
            fileType: fileType,
            Status: "S3 Download",
            Progress: "Step 3/5",
          });

          try {
            // Generate operation ID for progress tracking
            const operationId = `download_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            
            const downloadResponse = await fetch(
              `/api/download_dicom_from_s3?path=${encodeURIComponent(filePath)}&operation_id=${operationId}`,
              {
                signal: abortController.signal,
              },
            );

            if (!downloadResponse.ok) {
              const errorText = await downloadResponse.text();
              console.error("Download response error:", errorText);
              throw new Error(
                `Download failed: ${downloadResponse.status} ${downloadResponse.statusText}`,
              );
            }

            const dicomData = await downloadResponse.json();
            console.log(`[DEBUG] Full DICOM response:`, dicomData);
            

            console.log("DICOM data received:", dicomData);
            console.log(`[DEBUG] dicomData.dicom_file_path: ${dicomData.dicom_file_path}`);
            console.log(`[DEBUG] dicomData.cache_source: ${dicomData.cache_source}`);

            // Check for EX.DCM "WORK IN PROGRESS" response
            if (dicomData.message === "WORK IN PROGRESS" && dicomData.file_type === "ex_dcm") {
              console.log(`[EX.DCM] Work in progress detected for file: ${filePath}`);
              
              // Show "WORK IN PROGRESS" message in the viewport
              const img = document.getElementById(`viewportImage${viewportNumber}`);
              const placeholder = document.querySelector(`#viewportContent${viewportNumber} .viewport-placeholder`);
              const errorDiv = document.getElementById(`error${viewportNumber}`);
              
              // Hide other elements
              if (placeholder) placeholder.style.display = "none";
              if (img) img.style.display = "none";
              if (errorDiv) errorDiv.style.display = "none";
              
              // Show work in progress message
              showNotification("WORK IN PROGRESS - This file type is not yet supported", "error");
              
              // Update progress to show work in progress
              updateProgress(viewportNumber, 100, "WORK IN PROGRESS", {
                Status: "Not Supported",
                Type: "EX.DCM",
                Message: "Work in progress"
              });
              
              // Hide progress after a delay
              setTimeout(() => {
                hideProgress(viewportNumber);
              }, 3000);
              
              return; // Exit early, don't proceed with normal loading
            }
            
            // Check for FDS "WORK IN PROGRESS" response
            if (dicomData.message === "WORK IN PROGRESS" && dicomData.file_type === "fds") {
              console.log(`[FDS] Work in progress detected for file: ${filePath}`);
              
              // Show "WORK IN PROGRESS" message in the viewport
              const img = document.getElementById(`viewportImage${viewportNumber}`);
              const placeholder = document.querySelector(`#viewportContent${viewportNumber} .viewport-placeholder`);
              const errorDiv = document.getElementById(`error${viewportNumber}`);
              
              // Hide other elements
              if (placeholder) placeholder.style.display = "none";
              if (img) img.style.display = "none";
              if (errorDiv) errorDiv.style.display = "none";
              
              // Show work in progress message
              showNotification("WORK IN PROGRESS - This file type is not yet supported", "error");
              
              // Update progress to show work in progress
              updateProgress(viewportNumber, 100, "WORK IN PROGRESS", {
                Status: "Not Supported",
                Type: "FDS",
                Message: "Work in progress"
              });
              
              // Hide progress after a delay
              setTimeout(() => {
                hideProgress(viewportNumber);
              }, 3000);
              
              return; // Exit early, don't proceed with normal loading
            }

            // Check if cancelled after download
            if (abortController.signal.aborted) {
              throw new Error("Operation cancelled by user");
            }

            // Update cacheSource from response
            cacheSource = dicomData.cache_source || "fresh_download";
            let cacheMessage = "";
            switch (cacheSource) {
              case "memory":
                cacheMessage = "Loaded from memory cache";
                break;
              case "disk":
                cacheMessage = "Loaded from disk cache";
                break;
              case "fresh_download":
                cacheMessage = "Downloaded from S3";
                break;
              default:
                cacheMessage = "Processed";
            }

            // Store DICOM data for this viewport
            viewportData[viewportNumber] = dicomData;
            viewportData[viewportNumber].s3_key = filePath;

            // Step 4: Processing DICOM
            nextProgressStep(viewportNumber, cacheMessage, {
              Status: "Processing",
              Frames: dicomData.number_of_frames || 1,
              Cache: cacheSource.toUpperCase(),
              Progress: "Step 4/5",
            });

            // Step 5: Get frame information and load first frame
            nextProgressStep(viewportNumber, "Loading frame data...", {
              Status: "Frame Analysis",
              Cache: cacheSource.toUpperCase(),
              Progress: "Step 5/5",
            });

            const framesResponse = await fetch(
              `/api/view_frames/${dicomData.dicom_file_path}`,
              {
                signal: abortController.signal,
              },
            );

            if (!framesResponse.ok) {
              throw new Error(
                `Failed to get frame info: ${framesResponse.statusText}`,
              );
            }

            const framesData = await framesResponse.json();
            console.log("Frames data received:", framesData);

            // Check if cancelled after frame info
            if (abortController.signal.aborted) {
              clearInterval(progressInterval);
              progressManagers[viewportNumber].cleanupProgressPolling();
              throw new Error("Operation cancelled by user");
            }

            totalFrames[viewportNumber] = framesData.number_of_frames;
            currentFrames[viewportNumber] = 0;

            // Step 6: Auto image display logic
            const isOCT = isOCTDicom(viewportData[viewportNumber]);
            const fileType = detectFileType(filePath);
            console.log(`[DEBUG] File type detection: fileType=${fileType}, isOCT=${isOCT}, frames=${dicomData.number_of_frames}`);
            
            if (isOCT || fileType === 'FDA' || fileType === 'FDS') {
              // Handle OCT, FDA, and FDS files with flattening
              const processingType = fileType === 'FDA' ? 'FDA Processing' : 
                                   fileType === 'FDS' ? 'FDS Processing' : 'OCT Processing';
              
              nextProgressStep(viewportNumber, `Applying ${fileType} processing...`, {
                Status: processingType,
                Type: fileType,
                Cache: cacheSource.toUpperCase(),
                Progress: "Step 5/5",
              });

              if (fileType === 'E2E') {
                // E2E files have special handling
                console.log(`[DEBUG] Processing E2E file for viewport ${viewportNumber}`);
                // E2E files are handled differently - they load into the tree structure
                totalFrames[viewportNumber] = 1;
                currentFrames[viewportNumber] = 0;
              } else {
                // OCT, FDA, FDS files use flattening
                console.log(`[DEBUG] Processing ${fileType} file with flattening for viewport ${viewportNumber}`);
                await flattenImageDirectly(viewportNumber);
                
                // Override frame data to show only flattened version
                totalFrames[viewportNumber] = 1;
                currentFrames[viewportNumber] = 0;
              }
            } else {
              // Load the first frame with CRC caching for standard DICOM images
              console.log(`[DEBUG] Loading standard DICOM frame 0 for viewport ${viewportNumber}`);
              
              // Debug: Check DICOM file status before loading
              await checkDicomFileStatus(dicomData.dicom_file_path);
              
              await loadFrameWithCRC(viewportNumber, 0, abortController);

              nextProgressStep(viewportNumber, "Finalizing...", {
                Status: "Complete",
                Type: "Standard DICOM",
                Cache: cacheSource.toUpperCase(),
                Progress: "Step 5/5",
              });
            }
          } catch (error) {
            if (abortController.signal.aborted) {
              throw new Error("Operation cancelled by user");
            }
            console.error("Fetch error details:", error);
            
            // If it's a cache-related error, try clearing cache and retrying once
            if (error.message.includes("404") || error.message.includes("not found") || error.message.includes("empty") || 
                error.message.includes("not supported") || error.message.includes("failed")) {
              console.log("[RETRY] Attempting cache clear and retry due to error:", error.message);
              try {
                await clearCacheAndRetry(viewportNumber, filePath);
                return; // Success, exit early
              } catch (retryError) {
                console.error("[RETRY] Cache clear and retry failed:", retryError);
                throw retryError; // Re-throw the original error
              }
            }
            
            throw error;
          }

          // Setup frame slider
          setupFrameSlider(viewportNumber);

          // Note: Eye tree population is only needed for E2E files
          // Regular DICOM files don't require eye tree structure

          // Complete progress
          updateProgress(viewportNumber, 100, "Loading complete!", {
            Status: "Ready",
            Cache: cacheSource.toUpperCase(),
            Loaded: "Success",
          });

          // Hide progress after a brief delay
          setTimeout(() => {
            hideProgress(viewportNumber);
          }, 1000);

          // End performance timer
          endPerformanceTimer(viewportNumber, "File Selection â†’ Image Loaded");

          console.log(
            `Successfully loaded DICOM into viewport ${viewportNumber} from ${cacheSource}`,
          );
        } catch (error) {
          console.error("Error loading DICOM:", error);

          // Clean up progress polling
          if (progressManagers[viewportNumber]) {
            progressManagers[viewportNumber].cleanupProgressPolling();
          }

          if (error.message.includes("cancelled")) {
            progressManagers[viewportNumber].updateProgress(
              0,
              "Cancelled by user",
            );
            setTimeout(() => {
              hideProgress(viewportNumber);
              if (placeholder) placeholder.style.display = "block";
            }, 1000);
          } else {
            progressManagers[viewportNumber].setError(
              `Error: ${error.message}`,
            );

            setTimeout(() => {
              hideProgress(viewportNumber);
              errorDiv.textContent = `Error: ${error.message}`;
              errorDiv.style.display = "block";
              if (placeholder) placeholder.style.display = "block";
            }, 2000);
          }

          // End performance timer on error
          endPerformanceTimer(
            viewportNumber,
            "File Selection â†’ Image Loaded (ERROR)",
          );
        } finally {
          
        }
      }, 300); // 300ms debounce delay

      // Tree structure sorting functions
      function extractSortKey(itemName) {
        if (!itemName) return "";

        const itemUpper = itemName.toUpperCase();

        // Check for SCR
        if (itemUpper.includes("SCR")) {
          return "SCR";
        }

        // Check for Day patterns
        const dayMatch = itemUpper.match(/DAY\s*(\d+)/);
        if (dayMatch) {
          const dayNum = parseInt(dayMatch[1]);
          return `Day ${dayNum}`;
        }

        // Check for Week patterns
        const weekMatch = itemUpper.match(/WEEK\s*(\d+)/);
        if (weekMatch) {
          const weekNum = parseInt(weekMatch[1]);
          return `Week ${weekMatch[1]}`;
        }

        return itemName;
      }

      function customSortKey(item) {
        const name = item && item.name ? item.name : String(item);
        const sortKey = extractSortKey(name);

        // Get position in sort order
        try {
          const position = SORT_ORDER.indexOf(sortKey);
          return position !== -1
            ? [0, position, sortKey]
            : [1, 0, sortKey.toLowerCase()];
        } catch (error) {
          return [1, 0, sortKey.toLowerCase()];
        }
      }

      let currentSortMode = 'az';

      function sortTreeStructure(items) {
        try {
          // Separate folders and files
          const folders = items.filter((item) => item.type === "folder");
          const files = items.filter((item) => item.type !== "folder");

          // Sort folders by name A–Z
          const foldersSorted = folders.sort((a, b) => {
            const aName = (a.name || '').toLowerCase();
            const bName = (b.name || '').toLowerCase();
            return aName.localeCompare(bName);
          });

          // Sort files based on currentSortMode
          let filesSorted;
          if (currentSortMode === 'date') {
            filesSorted = files.sort((a, b) => {
              const aDate = a.last_modified ? new Date(a.last_modified).getTime() : 0;
              const bDate = b.last_modified ? new Date(b.last_modified).getTime() : 0;
              // Newest first
              return bDate - aDate;
            });
          } else if (currentSortMode === 'size') {
            filesSorted = files.sort((a, b) => {
              const aSize = a.size || 0;
              const bSize = b.size || 0;
              // Largest first
              return bSize - aSize;
            });
          } else if (currentSortMode === 'az') {
            filesSorted = files.sort((a, b) => {
              const aName = (a.name || '').toLowerCase();
              const bName = (b.name || '').toLowerCase();
              return aName.localeCompare(bName);
            });
          } else {
            // Fallback to custom key sorting if an unknown mode is set
            filesSorted = files.sort((a, b) => {
              const aKey = customSortKey(a);
              const bKey = customSortKey(b);
              for (let i = 0; i < Math.max(aKey.length, bKey.length); i++) {
                if (aKey[i] === undefined) return -1;
                if (bKey[i] === undefined) return 1;
                if (aKey[i] !== bKey[i]) {
                  return typeof aKey[i] === "string" ? aKey[i].localeCompare(bKey[i]) : aKey[i] - bKey[i];
                }
              }
              return 0;
            });
          }

          // Recursively sort subfolders
          foldersSorted.forEach((folder) => {
            if (folder.children && folder.children.length > 0) {
              folder.children = sortTreeStructure(folder.children);
            }
          });

          // Return folders first, then files
          return [...foldersSorted, ...filesSorted];
        } catch (error) {
          console.error("Error sorting tree structure:", error);
          return items;
        }
      }

      function toggleSortMenu() {
        const dropdownMenu = document.getElementById('sortDropdownMenu');
        const burgerButton = document.getElementById('sortBurgerButton');
        if (dropdownMenu && burgerButton) {
          const isVisible = dropdownMenu.classList.contains('show');
          if (isVisible) {
            dropdownMenu.classList.remove('show');
            burgerButton.classList.remove('active');
            burgerButton.setAttribute('aria-expanded', 'false');
          } else {
            dropdownMenu.classList.add('show');
            burgerButton.classList.add('active');
            burgerButton.setAttribute('aria-expanded', 'true');
          }
        }
      }

      function setSortMode(mode) {
        currentSortMode = mode;
        const dropdownMenu = document.getElementById('sortDropdownMenu');
        const burgerButton = document.getElementById('sortBurgerButton');
        if (dropdownMenu) dropdownMenu.classList.remove('show');
        if (burgerButton) burgerButton.classList.remove('active');

        // If a search or filter is active, re-run search; otherwise render current level
        const searchInput = document.getElementById('treeSearchInput');
        const extensionFilter = document.getElementById('extensionFilter');
        const hasSearch = searchInput && searchInput.value.trim();
        const hasFilter = extensionFilter && extensionFilter.value;

        if (hasSearch || hasFilter) {
          runTreeSearch();
        } else if (window.s3Browser && typeof window.s3Browser.renderCurrentLevel === 'function') {
          window.s3Browser.renderCurrentLevel();
        }
      }
      window.toggleSortMenu = toggleSortMenu;
      window.setSortMode = setSortMode;

      // Fit-to-viewport: set --header-height dynamically based on actual header size
      function applyHeaderHeightVar() {
        const header = document.querySelector('.custom-header');
        if (!header) return;
        const h = Math.round(header.getBoundingClientRect().height);
        document.documentElement.style.setProperty('--header-height', h + 'px');
      }
      
      // Bind dropdown triggers, keyboard access, and outside-click close
      function bindDropdowns() {
        const sortButton = document.getElementById('sortBurgerButton');
        const sortMenu = document.getElementById('sortDropdownMenu');
        const eyeButton = document.getElementById('eyeFocusBurgerButton');
        const eyeMenu = document.getElementById('eyeFocusDropdownMenu');
        
        // Ensure ARIA defaults
        if (sortButton) {
          sortButton.setAttribute('role', 'button');
          sortButton.setAttribute('aria-haspopup', 'true');
          sortButton.setAttribute('aria-expanded', sortMenu && sortMenu.classList.contains('show') ? 'true' : 'false');
          // Keyboard toggle for cross-browser accessibility
          sortButton.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              if (typeof window.toggleSortMenu === 'function') window.toggleSortMenu();
            }
          });
        }
        
        if (eyeButton) {
          eyeButton.setAttribute('role', 'button');
          eyeButton.setAttribute('aria-haspopup', 'true');
          eyeButton.setAttribute('aria-expanded', eyeMenu && eyeMenu.classList.contains('show') ? 'true' : 'false');
          eyeButton.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              if (typeof window.toggleEyeFocusMenu === 'function') window.toggleEyeFocusMenu();
            }
          });
        }
        
        // Keyboard activation on dropdown items
        if (sortMenu) {
          sortMenu.querySelectorAll('.eye-focus-item').forEach((item) => {
            item.setAttribute('tabindex', '0');
            item.setAttribute('role', 'menuitem');
            item.addEventListener('keydown', (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                const text = item.textContent || '';
                if (/A–Z/i.test(text)) window.setSortMode && window.setSortMode('az');
                else if (/By Date/i.test(text)) window.setSortMode && window.setSortMode('date');
                else if (/By Size/i.test(text)) window.setSortMode && window.setSortMode('size');
              }
            });
          });
        }
        
        if (eyeMenu) {
          const items = eyeMenu.querySelectorAll('.eye-focus-item');
          items.forEach((item) => {
            item.setAttribute('tabindex', '0');
            item.setAttribute('role', 'menuitem');
            item.addEventListener('keydown', (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                const txt = (item.textContent || '').toLowerCase();
                if (txt.includes('left')) window.focusOnEye && window.focusOnEye('right');
                else if (txt.includes('right')) window.focusOnEye && window.focusOnEye('left');
              }
            });
          });
        }
        
        // Outside click closes both menus
        document.addEventListener('mousedown', handleGlobalDropdownClose, true);
        document.addEventListener('touchstart', handleGlobalDropdownClose, { passive: true, capture: true });
        // Escape closes menus
        document.addEventListener('keyup', (e) => {
          if (e.key === 'Escape') {
            const sBtn = document.getElementById('sortBurgerButton');
            const sMenu = document.getElementById('sortDropdownMenu');
            const eBtn = document.getElementById('eyeFocusBurgerButton');
            const eMenu = document.getElementById('eyeFocusDropdownMenu');
            if (sMenu && sMenu.classList.contains('show')) {
              sMenu.classList.remove('show');
              sBtn && sBtn.setAttribute('aria-expanded', 'false');
              sBtn && sBtn.classList.remove('active');
            }
            if (eMenu && eMenu.classList.contains('show')) {
              eMenu.classList.remove('show');
              eBtn && eBtn.setAttribute('aria-expanded', 'false');
              eBtn && eBtn.classList.remove('active');
            }
          }
        });
      }
      
      function handleGlobalDropdownClose(ev) {
        const sortButton = document.getElementById('sortBurgerButton');
        const sortMenu = document.getElementById('sortDropdownMenu');
        const eyeButton = document.getElementById('eyeFocusBurgerButton');
        const eyeMenu = document.getElementById('eyeFocusDropdownMenu');
        const t = ev.target;
        // Close sort if clicking outside
        if (sortMenu && !sortMenu.contains(t) && sortButton && !sortButton.contains(t)) {
          sortMenu.classList.remove('show');
          sortButton.classList.remove('active');
          sortButton.setAttribute('aria-expanded', 'false');
        }
        // Close eye focus if clicking outside
        if (eyeMenu && !eyeMenu.contains(t) && eyeButton && !eyeButton.contains(t)) {
          eyeMenu.classList.remove('show');
          eyeButton.classList.remove('active');
          eyeButton.setAttribute('aria-expanded', 'false');
        }
      }
      
      window.addEventListener('load', applyHeaderHeightVar);
      window.addEventListener('resize', applyHeaderHeightVar);
      window.addEventListener('DOMContentLoaded', bindDropdowns);

      // CRC-based Image Cache System
      class CRCImageCache {
        constructor() {
          this.cache = new Map();
          this.maxCacheSize = 100; // Maximum number of cached images
          this.cacheStats = {
            hits: 0,
            misses: 0,
            evictions: 0,
          };
        }

        // Calculate CRC32 for a given string/buffer
        calculateCRC32(data) {
          const crcTable = new Uint32Array(256);
          for (let i = 0; i < 256; i++) {
            let c = i;
            for (let j = 0; j < 8; j++) {
              c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
            }
            crcTable[i] = c;
          }

          let crc = 0 ^ -1;
          const bytes = new TextEncoder().encode(data);
          for (let i = 0; i < bytes.length; i++) {
            crc = (crc >>> 8) ^ crcTable[(crc ^ bytes[i]) & 0xff];
          }
          return ((crc ^ -1) >>> 0).toString(16).padStart(8, "0");
        }

        // Generate cache key from file path and metadata
        generateCacheKey(filePath, metadata = {}) {
          const keyData = JSON.stringify({
            path: filePath,
            size: metadata.size || 0,
            lastModified: metadata.lastModified || "",
            frame: metadata.frame || 0,
          });
          return this.calculateCRC32(keyData);
        }

        // Check if image exists in cache
        has(cacheKey) {
          return this.cache.has(cacheKey);
        }

        // Get image from cache
        get(cacheKey) {
          if (this.cache.has(cacheKey)) {
            const item = this.cache.get(cacheKey);
            // Update access time for LRU
            item.lastAccessed = Date.now();
            this.cacheStats.hits++;
            console.log(`[CACHE HIT] Key: ${cacheKey}`);
            return item;
          }
          this.cacheStats.misses++;
          console.log(`[CACHE MISS] Key: ${cacheKey}`);
          return null;
        }

        // Store image in cache
        set(cacheKey, imageData, metadata = {}) {
          // Evict oldest items if cache is full
          if (this.cache.size >= this.maxCacheSize) {
            this.evictOldest();
          }

          const cacheItem = {
            imageData: imageData,
            metadata: metadata,
            cachedAt: Date.now(),
            lastAccessed: Date.now(),
            cacheKey: cacheKey,
          };

          this.cache.set(cacheKey, cacheItem);
          console.log(`[CACHE SET] Key: ${cacheKey}, Size: ${this.cache.size}`);
        }

        // Evict oldest cache entry
        evictOldest() {
          let oldestKey = null;
          let oldestTime = Date.now();

          for (const [key, item] of this.cache.entries()) {
            if (item.lastAccessed < oldestTime) {
              oldestTime = item.lastAccessed;
              oldestKey = key;
            }
          }

          if (oldestKey) {
            this.cache.delete(oldestKey);
            this.cacheStats.evictions++;
            console.log(`[CACHE EVICT] Key: ${oldestKey}`);
          }
        }

        // Get cache statistics
        getStats() {
          return {
            ...this.cacheStats,
            size: this.cache.size,
            maxSize: this.maxCacheSize,
            hitRate:
              this.cacheStats.hits /
                (this.cacheStats.hits + this.cacheStats.misses) || 0,
          };
        }

        // Clear cache
        clear() {
          this.cache.clear();
          this.cacheStats = { hits: 0, misses: 0, evictions: 0 };
          console.log("[CACHE CLEAR] Cache cleared");
        }
      }

      // Initialize global cache instance
      const imageCache = new CRCImageCache();

      // Enhanced image loading with CRC-based caching and OCT flattening priority
      async function loadImageWithCRC(filePath, metadata = {}) {
        try {
          // For OCT images, check for flattened version first
          if (metadata.isOCT || filePath.toLowerCase().includes("oct")) {
            const flattenMetadata = {
              path: filePath,
              flattened: true,
              frame: metadata.frame || 0,
            };

            const flattenCacheKey = imageCache.generateCacheKey(
              filePath + "_flattened",
              flattenMetadata,
            );
            const cachedFlattened = imageCache.get(flattenCacheKey);

            if (cachedFlattened) {
              console.log(
                `[OCT CACHE] Using cached flattened image for ${filePath}`,
              );
              return {
                imageData: cachedFlattened.imageData,
                source: "cache_flattened",
                cacheKey: flattenCacheKey,
                metadata: cachedFlattened.metadata,
              };
            }
          }

          // Step 1: Generate CRC-based cache key for original image
          const cacheKey = imageCache.generateCacheKey(filePath, metadata);
          console.log(`[CRC CACHE] Generated key: ${cacheKey} for ${filePath}`);

          // Step 2: Check local cache for original image (only for non-OCT)
          if (!metadata.isOCT && !filePath.toLowerCase().includes("oct")) {
            const cachedItem = imageCache.get(cacheKey);
            if (cachedItem) {
              console.log(`[CRC CACHE] Using cached image for ${filePath}`);
              return {
                imageData: cachedItem.imageData,
                source: "cache",
                cacheKey: cacheKey,
                metadata: cachedItem.metadata,
              };
            }
          }

          // Continue with rest of the function...
          // (Keep the existing backend CRC check and download logic)
          // Step 3: Check if backend has CRC info
          let backendCRC = null;
          try {
            const crcResponse = await fetch(
              `/api/get-file-crc?path=${encodeURIComponent(filePath)}`,
            );
            if (crcResponse.ok) {
              const crcData = await crcResponse.json();
              backendCRC = crcData.crc;

              // If backend CRC differs from our calculated CRC, use backend CRC
              if (backendCRC && backendCRC !== cacheKey) {
                const backendCachedItem = imageCache.get(backendCRC);
                if (backendCachedItem) {
                  console.log(
                    `[CRC CACHE] Using backend CRC cached image: ${backendCRC}`,
                  );
                  return {
                    imageData: backendCachedItem.imageData,
                    source: "cache_backend_crc",
                    cacheKey: backendCRC,
                    metadata: backendCachedItem.metadata,
                  };
                }
              }
            }
          } catch (error) {
            console.warn(
              `[CRC CACHE] Could not get backend CRC: ${error.message}`,
            );
          }

          // Step 4: Download image with CRC query parameter for HTTP caching
          const finalCacheKey = backendCRC || cacheKey;
          
          // Use the correct dicom_file_path - prioritize metadata.dicomFilePath over filePath
          const dicomFilePath = metadata.dicomFilePath || filePath;
          console.log(`[CRC CACHE] Using dicom_file_path: ${dicomFilePath} for file: ${filePath}`);
          
          const imageUrl = `/api/view_dicom_png?frame=${metadata.frame || 0}&dicom_file_path=${encodeURIComponent(dicomFilePath)}&v=${finalCacheKey}`;

          console.log(
            `[CRC CACHE] Downloading image with CRC: ${finalCacheKey}`,
          );
          console.log(`[CRC CACHE] Image URL: ${imageUrl}`);
          
          const response = await fetch(imageUrl, {
            headers: {
              "Cache-Control": "public, max-age=31536000, immutable",
            },
          });

          if (!response.ok) {
            const errorText = await response.text().catch(() => "Unknown error");
            console.error(`[CRC CACHE] HTTP Error ${response.status}: ${response.statusText}`);
            console.error(`[CRC CACHE] Error details: ${errorText}`);
            throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
          }

          const imageBlob = await response.blob();
          console.log(`[CRC CACHE] Downloaded image blob: ${imageBlob.size} bytes, type: ${imageBlob.type}`);
          
          if (imageBlob.size === 0) {
            throw new Error("Downloaded image is empty (0 bytes)");
          }
          
          const imageUrl_cached = URL.createObjectURL(imageBlob);

          // Step 5: Store in cache
          const cacheMetadata = {
            ...metadata,
            downloadedAt: Date.now(),
            contentType: response.headers.get("content-type"),
            size: imageBlob.size,
          };

          imageCache.set(finalCacheKey, imageUrl_cached, cacheMetadata);

          return {
            imageData: imageUrl_cached,
            source: "download",
            cacheKey: finalCacheKey,
            metadata: cacheMetadata,
          };
        } catch (error) {
          console.error(`[CRC CACHE] Error loading image: ${error.message}`);
          throw error;
        }
      }

      // Auto image display logic - determine if DICOM is OCT
      // Enhanced OCT detection
      function isOCTDicom(dicomData) {
        if (!dicomData) return false;

        // Check filename patterns
        if (dicomData.s3_key) {
          const filename = dicomData.s3_key.toLowerCase();
          if (filename.includes("oct") || filename.includes("optical")) {
            return true;
          }
        }

        // Check if it's multi-frame (common for OCT)
        if (dicomData.number_of_frames && dicomData.number_of_frames > 1) {
          return true;
        }

        return false;
      }

      // Enhanced file type detection for better debugging
      function detectFileType(filePath) {
        const filename = filePath.toLowerCase();
        if (filename.endsWith('.dcm') || filename.endsWith('.dicom')) {
          return 'DICOM';
        } else if (filename.endsWith('.e2e')) {
          return 'E2E';
        } else if (filename.endsWith('.fda')) {
          return 'FDA';
        } else if (filename.endsWith('.fds')) {
          return 'FDS';
        } else {
          return 'UNKNOWN';
        }
      }

      // Function to validate file type support
      function validateFileType(filePath) {
        const fileType = detectFileType(filePath);
        const supportedTypes = ['DICOM', 'E2E', 'FDA', 'FDS'];
        
        if (!supportedTypes.includes(fileType)) {
          throw new Error(`Unsupported file type: ${fileType}. Supported types: ${supportedTypes.join(', ')}`);
        }
        
        return fileType;
      }
      function isOCT(dicomData) {
        return isOCTDicom(dicomData);
      }

      // Performance monitoring functions
      function startPerformanceTimer(viewportNumber, operation) {
        performanceTimers[viewportNumber] = {
          start: performance.now(),
          operation: operation,
        };
        updatePerformanceDisplay(`${operation} started...`);
      }

      function endPerformanceTimer(viewportNumber, operation) {
        if (performanceTimers[viewportNumber]) {
          const elapsed =
            performance.now() - performanceTimers[viewportNumber].start;
          const elapsedSeconds = (elapsed / 1000).toFixed(2);
          updatePerformanceDisplay(
            `${operation} completed in ${elapsedSeconds}s`,
          );

          // Log to console for debugging
          console.log(
            `[PERFORMANCE] Viewport ${viewportNumber} - ${operation}: ${elapsedSeconds}s`,
          );

          // Warn if over 7 seconds
          if (elapsed > 7000) {
            console.warn(
              `[PERFORMANCE WARNING] Operation took ${elapsedSeconds}s (target: <7s)`,
            );
          }

          performanceTimers[viewportNumber] = null;
        }
      }

      function updatePerformanceDisplay(message) {
        const monitor = document.getElementById("performanceMonitor");
        if (monitor) {
          monitor.textContent = `Performance: ${message}`;
        }
      }

      // DICOM Layout toggle functionality
      function toggleDicomViewportLayout() {
        console.log("toggleDicomViewportLayout called");
        const container = document.getElementById("dicomViewportsContainer");
        const icon = document.getElementById("dicomLayoutIcon");
        const text = document.getElementById("dicomLayoutText");

        console.log("Elements found:", { container, icon, text });

        isDicomStackedLayout = !isDicomStackedLayout;

        if (isDicomStackedLayout) {
          container.classList.remove("side-by-side");
          container.classList.add("stacked");
          icon.className = "fas fa-columns";
          text.textContent = "Switch to Side-by-Side";
        } else {
          container.classList.remove("stacked");
          container.classList.add("side-by-side");
          icon.className = "fas fa-grip-lines";
          text.textContent = "Switch to Stacked";
        }

        console.log(
          `DICOM viewport layout changed to: ${isDicomStackedLayout ? "stacked" : "side-by-side"}`,
        );
      }

      // Update the existing S3 layout toggle to be more specific
      function toggleViewportLayout() {
        // Don't allow layout toggle in focus mode
        if (focusedEye) {
          console.log("Layout toggle disabled in focus mode");
          return;
        }

        const container = document.getElementById("viewportsContainer");
        const icon = document.getElementById("layoutIcon");
        const text = document.getElementById("layoutText");

        isStackedLayout = !isStackedLayout;

        // Remove any previous wrappers
        function removeRowWrappers() {
          const leftRow = document.getElementById("leftViewportRow");
          const rightRow = document.getElementById("rightViewportRow");
          if (leftRow) {
            if (viewport1 && leftRow.contains(viewport1)) container.appendChild(viewport1);
            if (leftTreeContainer && leftRow.contains(leftTreeContainer)) viewport1.appendChild(leftTreeContainer);
            leftRow.remove();
          }
          if (rightRow) {
            if (viewport2 && rightRow.contains(viewport2)) container.appendChild(viewport2);
            if (rightTreeContainer && rightRow.contains(rightTreeContainer)) viewport2.appendChild(rightTreeContainer);
            rightRow.remove();
          }
        }

        if (isStackedLayout) {
          container.classList.remove("side-by-side");
          container.style.flexDirection = "column";
          container.style.gap = "8px";
          icon.className = "fas fa-grip-lines";
          text.textContent = "Switch to Side-by-Side";

          // E2E mode: each viewport + its tree in a row (80/20)
          if (isE2EMode) {
            removeRowWrappers();
            // LEFT
            const leftRow = document.createElement("div");
            leftRow.id = "leftViewportRow";
            leftRow.style.display = "flex";
            leftRow.style.flexDirection = "row";
            leftRow.style.width = "100%";
            leftRow.style.height = "50%";
            leftRow.style.minHeight = "calc((100vh - 200px) / 2)";
            leftRow.style.gap = "8px";
            leftRow.style.alignItems = "stretch";
            // Set viewport1 styles
            if (viewport1) {
              viewport1.style.width = "80%";
              viewport1.style.flex = "0 0 80%";
              viewport1.style.height = "100%";
              viewport1.style.minHeight = "calc((100vh - 200px) / 2)";
              leftRow.appendChild(viewport1);
            }
            if (leftTreeContainer) {
              leftTreeContainer.style.display = "block";
              leftTreeContainer.style.width = "20%";
              leftTreeContainer.style.height = "100%";
              leftTreeContainer.style.minHeight = "calc((100vh - 200px) / 2)";
              leftTreeContainer.style.flex = "0 0 20%";
              leftTreeContainer.style.marginBottom = "0";
              leftTreeContainer.style.marginTop = "0";
              leftTreeContainer.style.border = "1px solid #e9ecef";
              leftRow.appendChild(leftTreeContainer);
            }
            container.appendChild(leftRow);
            // RIGHT
            const rightRow = document.createElement("div");
            rightRow.id = "rightViewportRow";
            rightRow.style.display = "flex";
            rightRow.style.flexDirection = "row";
            rightRow.style.width = "100%";
            rightRow.style.height = "50%";
            rightRow.style.minHeight = "calc((100vh - 200px) / 2)";
            rightRow.style.gap = "8px";
            rightRow.style.alignItems = "stretch";
            if (viewport2) {
              viewport2.style.width = "80%";
              viewport2.style.flex = "0 0 80%";
              viewport2.style.height = "100%";
              viewport2.style.minHeight = "calc((100vh - 200px) / 2)";
              rightRow.appendChild(viewport2);
            }
            if (rightTreeContainer) {
              rightTreeContainer.style.display = "block";
              rightTreeContainer.style.width = "20%";
              rightTreeContainer.style.height = "100%";
              rightTreeContainer.style.minHeight = "calc((100vh - 200px) / 2)";
              rightTreeContainer.style.flex = "0 0 20%";
              rightTreeContainer.style.marginBottom = "0";
              rightTreeContainer.style.maxHeight = "100%";
              rightTreeContainer.style.marginTop = "0";
              rightTreeContainer.style.border = "1px solid #e9ecef";
              rightRow.appendChild(rightTreeContainer);
            }
            container.appendChild(rightRow);
          } else {
            // For non-E2E mode, use original stacked behavior
            removeRowWrappers();
            if (viewport1) {
              viewport1.style.width = "100%";
              viewport1.style.flex = "1";
              viewport1.style.height = "50%";
              viewport1.style.minHeight = "calc((100vh - 200px) / 2)";
            }
            if (viewport2) {
              viewport2.style.width = "100%";
              viewport2.style.flex = "1";
              viewport2.style.height = "50%";
              viewport2.style.minHeight = "calc((100vh - 200px) / 2)";
            }
            if (leftTreeContainer) leftTreeContainer.style.display = "none";
            if (rightTreeContainer) rightTreeContainer.style.display = "none";
          }
        } else {
          container.classList.add("side-by-side");
          container.style.flexDirection = "row";
          icon.className = "fas fa-columns";
          text.textContent = "Switch to Stacked";

          // Clean up E2E stacked wrappers if present
          removeRowWrappers();

          // Reset viewport styles
          if (viewport1) {
            viewport1.style.width = "";
            viewport1.style.flex = "1";
          }
          if (viewport2) {
            viewport2.style.width = "";
            viewport2.style.flex = "1";
          }

          // Show eye trees in side-by-side layout if E2E
          if (leftTreeContainer && isE2EMode) {
            leftTreeContainer.style.display = "block";
            leftTreeContainer.style.width = "";
            leftTreeContainer.style.flex = "";
            leftTreeContainer.style.border = "2px solid #e9ecef";
            leftTreeContainer.style.marginBottom = "";
            leftTreeContainer.style.maxHeight = "300px";
            if (viewport1 && !viewport1.contains(leftTreeContainer)) {
              viewport1.appendChild(leftTreeContainer);
            }
          }
          if (rightTreeContainer && isE2EMode) {
            rightTreeContainer.style.display = "block";
            rightTreeContainer.style.width = "";
            rightTreeContainer.style.flex = "";
            rightTreeContainer.style.border = "2px solid #e9ecef";
            rightTreeContainer.style.maxHeight = "300px";
            if (viewport2 && !viewport2.contains(rightTreeContainer)) {
              viewport2.appendChild(rightTreeContainer);
            }
          }
        }

        console.log(
          `S3 viewport layout changed to: ${isStackedLayout ? "stacked" : "side-by-side"}`,
        );
        // Ensure eye trees remain visible in E2E mode after layout change
        if (isE2EMode) {
          try { showE2EControls(); } catch (e) { console.warn('showE2EControls not available', e); }
        }
      }

      // Enhanced E2E layout toggle function with proper tree positioning
      function toggleViewportLayoutSimple() {
        // Don't allow layout toggle in focus mode
        if (focusedEye) {
          console.log("Layout toggle disabled in focus mode");
          return;
        }

        const container = document.getElementById("viewportsContainer");
        const icon = document.getElementById("layoutIcon");
        const text = document.getElementById("layoutText");
        const leftTreeContainer = document.getElementById("leftEyeTreeContainer");
        const rightTreeContainer = document.getElementById("rightEyeTreeContainer");
        const viewport1 = document.getElementById("viewport1");
        const viewport2 = document.getElementById("viewport2");

        isStackedLayout = !isStackedLayout;

        if (isStackedLayout) {
          container.classList.remove("side-by-side");
          icon.className = "fas fa-grip-lines";
          text.textContent = "Switch to Side-by-Side";

          // Non-E2E stacked mode: Use default CSS grid (1 column, 2 rows)
          if (!isE2EMode) {
            // Just ensure side-by-side class is removed - default CSS handles stacked layout
            container.style.gridTemplateColumns = '1fr';
            container.style.gridTemplateRows = '1fr 1fr';
            container.style.gap = '10px';
            
            // Reset any inline styles on viewports that might interfere
            if (viewport1) {
              viewport1.style.width = '';
              viewport1.style.height = '';
              viewport1.style.flex = '';
            }
            if (viewport2) {
              viewport2.style.width = '';
              viewport2.style.height = '';
              viewport2.style.flex = '';
            }
          }

          // E2E stacked mode: Viewports with compact controls at bottom, trees on right
          if (isE2EMode) {
            // Remove any existing wrappers first
            const existingLeftRow = document.getElementById("leftViewportRow");
            const existingRightRow = document.getElementById("rightViewportRow");
            const existingViewportColumn = document.getElementById("e2eViewportColumn");
            const existingTreeColumn = document.getElementById("e2eTreeColumn");
            const existingControlsColumn = document.getElementById("e2eControlsColumn");
            if (existingLeftRow) existingLeftRow.remove();
            if (existingRightRow) existingRightRow.remove();
            if (existingViewportColumn) existingViewportColumn.remove();
            if (existingTreeColumn) existingTreeColumn.remove();
            if (existingControlsColumn) existingControlsColumn.remove();

            // Get control elements
            const frameSlider1 = document.getElementById("frameSliderContainer1");
            const modeSwitch1 = document.getElementById("modeSwitchControls1");
            const frameInput1 = document.getElementById("frameInputControls1");
            const frameSlider2 = document.getElementById("frameSliderContainer2");
            const modeSwitch2 = document.getElementById("modeSwitchControls2");
            const frameInput2 = document.getElementById("frameInputControls2");

            // Change container to flex row layout
            container.style.display = 'flex';
            container.style.flexDirection = 'row';
            container.style.gap = '10px';
            container.style.width = '100%';
            container.style.height = '100%';

            // Create left column: viewports with controls at bottom
            const viewportColumn = document.createElement("div");
            viewportColumn.id = "e2eViewportColumn";
            viewportColumn.style.cssText = `
              display: flex;
              flex-direction: column;
              flex: 0 0 70%;
              width: 70%;
              gap: 10px;
              height: 100%;
            `;

            // Create wrapper for viewport 1 with compact controls
            const viewport1Wrapper = document.createElement("div");
            viewport1Wrapper.style.cssText = `
              display: flex;
              flex-direction: column;
              flex: 1;
              min-height: 0;
              height: 50%;
            `;

            // Set viewport1 styles (keep controls inside but make them compact)
            if (viewport1) {
              viewport1.style.cssText = `
                flex: 1;
                width: 100%;
                min-height: 0;
                overflow: hidden;
              `;
              viewport1Wrapper.appendChild(viewport1);
            }

            // Create compact controls bar for viewport 1
            const controlsBar1 = document.createElement("div");
            controlsBar1.style.cssText = `
              display: flex;
              flex-direction: row;
              align-items: center;
              gap: 8px;
              padding: 5px 10px;
              background: #f8f9fa;
              border-top: 1px solid #dee2e6;
              flex-shrink: 0;
            `;

            // Compact frame slider for viewport 1
            if (frameSlider1 && viewport1.contains(frameSlider1)) {
              viewport1.removeChild(frameSlider1);
              frameSlider1.style.cssText = `
                display: flex;
                flex-direction: row;
                align-items: center;
                gap: 5px;
                margin: 0;
                padding: 0;
                background: transparent;
                flex: 1;
              `;
              const sliderLabel = frameSlider1.querySelector('label');
              if (sliderLabel) sliderLabel.style.cssText = 'margin: 0 5px 0 0; font-size: 12px; white-space: nowrap;';
              const sliderControls = frameSlider1.querySelector('.frame-slider-controls');
              if (sliderControls) sliderControls.style.cssText = 'display: flex; align-items: center; gap: 5px; flex: 1;';
              const slider = frameSlider1.querySelector('.frame-slider');
              if (slider) slider.style.cssText = 'flex: 1; min-width: 100px;';
              const frameInfo = frameSlider1.querySelector('.frame-info');
              if (frameInfo) frameInfo.style.cssText = 'margin: 0 5px; font-size: 11px; white-space: nowrap;';
              controlsBar1.appendChild(frameSlider1);
            }

            // Compact mode switch for viewport 1
            if (modeSwitch1 && viewport1.contains(modeSwitch1)) {
              viewport1.removeChild(modeSwitch1);
              modeSwitch1.style.cssText = `
                display: flex;
                gap: 5px;
                margin: 0;
                padding: 0;
              `;
              const buttons = modeSwitch1.querySelectorAll('button');
              buttons.forEach(btn => {
                btn.style.cssText = 'padding: 4px 8px; font-size: 11px;';
              });
              controlsBar1.appendChild(modeSwitch1);
            }

            // Compact frame input for viewport 1
            if (frameInput1 && viewport1.contains(frameInput1)) {
              viewport1.removeChild(frameInput1);
              frameInput1.style.cssText = `
                display: flex;
                align-items: center;
                gap: 5px;
                margin: 0;
                padding: 0;
              `;
              const inputGroup = frameInput1.querySelector('.frame-input-group');
              if (inputGroup) {
                inputGroup.style.cssText = 'display: flex; align-items: center; gap: 5px;';
                const label = inputGroup.querySelector('label');
                if (label) label.style.cssText = 'margin: 0; font-size: 11px; white-space: nowrap;';
                const input = inputGroup.querySelector('input');
                if (input) input.style.cssText = 'width: 50px; padding: 2px 5px; font-size: 11px;';
                const goBtn = inputGroup.querySelector('button');
                if (goBtn) goBtn.style.cssText = 'padding: 2px 8px; font-size: 11px;';
              }
              controlsBar1.appendChild(frameInput1);
            }

            viewport1Wrapper.appendChild(controlsBar1);
            viewportColumn.appendChild(viewport1Wrapper);

            // Create wrapper for viewport 2 with compact controls
            const viewport2Wrapper = document.createElement("div");
            viewport2Wrapper.style.cssText = `
              display: flex;
              flex-direction: column;
              flex: 1;
              min-height: 0;
              height: 50%;
            `;

            // Set viewport2 styles (keep controls inside but make them compact)
            if (viewport2) {
              viewport2.style.cssText = `
                flex: 1;
                width: 100%;
                min-height: 0;
                overflow: hidden;
              `;
              viewport2Wrapper.appendChild(viewport2);
            }

            // Create compact controls bar for viewport 2
            const controlsBar2 = document.createElement("div");
            controlsBar2.style.cssText = `
              display: flex;
              flex-direction: row;
              align-items: center;
              gap: 8px;
              padding: 5px 10px;
              background: #f8f9fa;
              border-top: 1px solid #dee2e6;
              flex-shrink: 0;
            `;

            // Compact frame slider for viewport 2
            if (frameSlider2 && viewport2.contains(frameSlider2)) {
              viewport2.removeChild(frameSlider2);
              frameSlider2.style.cssText = `
                display: flex;
                flex-direction: row;
                align-items: center;
                gap: 5px;
                margin: 0;
                padding: 0;
                background: transparent;
                flex: 1;
              `;
              const sliderLabel = frameSlider2.querySelector('label');
              if (sliderLabel) sliderLabel.style.cssText = 'margin: 0 5px 0 0; font-size: 12px; white-space: nowrap;';
              const sliderControls = frameSlider2.querySelector('.frame-slider-controls');
              if (sliderControls) sliderControls.style.cssText = 'display: flex; align-items: center; gap: 5px; flex: 1;';
              const slider = frameSlider2.querySelector('.frame-slider');
              if (slider) slider.style.cssText = 'flex: 1; min-width: 100px;';
              const frameInfo = frameSlider2.querySelector('.frame-info');
              if (frameInfo) frameInfo.style.cssText = 'margin: 0 5px; font-size: 11px; white-space: nowrap;';
              controlsBar2.appendChild(frameSlider2);
            }

            // Compact mode switch for viewport 2
            if (modeSwitch2 && viewport2.contains(modeSwitch2)) {
              viewport2.removeChild(modeSwitch2);
              modeSwitch2.style.cssText = `
                display: flex;
                gap: 5px;
                margin: 0;
                padding: 0;
              `;
              const buttons = modeSwitch2.querySelectorAll('button');
              buttons.forEach(btn => {
                btn.style.cssText = 'padding: 4px 8px; font-size: 11px;';
              });
              controlsBar2.appendChild(modeSwitch2);
            }

            // Compact frame input for viewport 2
            if (frameInput2 && viewport2.contains(frameInput2)) {
              viewport2.removeChild(frameInput2);
              frameInput2.style.cssText = `
                display: flex;
                align-items: center;
                gap: 5px;
                margin: 0;
                padding: 0;
              `;
              const inputGroup = frameInput2.querySelector('.frame-input-group');
              if (inputGroup) {
                inputGroup.style.cssText = 'display: flex; align-items: center; gap: 5px;';
                const label = inputGroup.querySelector('label');
                if (label) label.style.cssText = 'margin: 0; font-size: 11px; white-space: nowrap;';
                const input = inputGroup.querySelector('input');
                if (input) input.style.cssText = 'width: 50px; padding: 2px 5px; font-size: 11px;';
                const goBtn = inputGroup.querySelector('button');
                if (goBtn) goBtn.style.cssText = 'padding: 2px 8px; font-size: 11px;';
              }
              controlsBar2.appendChild(frameInput2);
            }

            viewport2Wrapper.appendChild(controlsBar2);
            viewportColumn.appendChild(viewport2Wrapper);

            container.appendChild(viewportColumn);

            // Create right column: trees stacked vertically
            const treeColumn = document.createElement("div");
            treeColumn.id = "e2eTreeColumn";
            treeColumn.style.cssText = `
              display: flex;
              flex-direction: column;
              flex: 0 0 30%;
              width: 30%;
              gap: 10px;
              height: 100%;
            `;

            // Set left tree styles for stacked mode
            if (leftTreeContainer) {
              leftTreeContainer.style.cssText = `
                flex: 1;
                width: 100%;
                min-height: 0;
                height: 50%;
                display: block;
                border: 1px solid #e9ecef;
                background: #f8f9fa;
                overflow-y: auto;
                padding: 10px;
                box-sizing: border-box;
              `;
              treeColumn.appendChild(leftTreeContainer);
            }

            // Set right tree styles for stacked mode
            if (rightTreeContainer) {
              rightTreeContainer.style.cssText = `
                flex: 1;
                width: 100%;
                min-height: 0;
                height: 50%;
                display: block;
                border: 1px solid #e9ecef;
                background: #f8f9fa;
                overflow-y: auto;
                padding: 10px;
                box-sizing: border-box;
              `;
              treeColumn.appendChild(rightTreeContainer);
            }

            container.appendChild(treeColumn);
          }
        } else {
          container.classList.add("side-by-side");
          icon.className = "fas fa-columns";
          text.textContent = "Switch to Stacked";

          // Non-E2E side-by-side mode: Use CSS grid (2 columns, 1 row)
          if (!isE2EMode) {
            container.style.gridTemplateColumns = '1fr 1fr';
            container.style.gridTemplateRows = '1fr';
            container.style.gap = '10px';
            
            // Reset any inline styles on viewports
            if (viewport1) {
              viewport1.style.width = '';
              viewport1.style.height = '';
              viewport1.style.flex = '';
            }
            if (viewport2) {
              viewport2.style.width = '';
              viewport2.style.height = '';
              viewport2.style.flex = '';
            }
          }

          // E2E side-by-side mode: Clean up stacked wrappers and reset styles
          if (isE2EMode) {
            // Remove old row wrappers (if they exist from previous version)
            const leftRow = document.getElementById("leftViewportRow");
            const rightRow = document.getElementById("rightViewportRow");
            if (leftRow) {
              if (viewport1 && leftRow.contains(viewport1)) container.appendChild(viewport1);
              if (leftTreeContainer && leftRow.contains(leftTreeContainer)) viewport1.appendChild(leftTreeContainer);
              leftRow.remove();
            }
            if (rightRow) {
              if (viewport2 && rightRow.contains(viewport2)) container.appendChild(viewport2);
              if (rightTreeContainer && rightRow.contains(rightTreeContainer)) viewport2.appendChild(rightTreeContainer);
              rightRow.remove();
            }

            // Remove new column wrappers
            const viewportColumn = document.getElementById("e2eViewportColumn");
            const treeColumn = document.getElementById("e2eTreeColumn");
            
            // Get control elements to restore
            const frameSlider1 = document.getElementById("frameSliderContainer1");
            const modeSwitch1 = document.getElementById("modeSwitchControls1");
            const frameInput1 = document.getElementById("frameInputControls1");
            const frameSlider2 = document.getElementById("frameSliderContainer2");
            const modeSwitch2 = document.getElementById("modeSwitchControls2");
            const frameInput2 = document.getElementById("frameInputControls2");
            
            if (viewportColumn) {
              // Find viewport wrappers and extract viewports and controls
              const viewport1Wrapper = viewportColumn.querySelector('div:first-child');
              const viewport2Wrapper = viewportColumn.querySelector('div:last-child');
              
              if (viewport1Wrapper) {
                // Extract viewport1
                if (viewport1 && viewport1Wrapper.contains(viewport1)) {
                  container.appendChild(viewport1);
                }
                // Extract controls from control bar
                const controlsBar1 = viewport1Wrapper.querySelector('div:last-child');
                if (controlsBar1) {
                  if (frameSlider1 && controlsBar1.contains(frameSlider1) && viewport1) {
                    controlsBar1.removeChild(frameSlider1);
                    viewport1.appendChild(frameSlider1);
                    frameSlider1.style.cssText = '';
                    // Reset child styles
                    const sliderLabel = frameSlider1.querySelector('label');
                    if (sliderLabel) sliderLabel.style.cssText = '';
                    const sliderControls = frameSlider1.querySelector('.frame-slider-controls');
                    if (sliderControls) sliderControls.style.cssText = '';
                    const slider = frameSlider1.querySelector('.frame-slider');
                    if (slider) slider.style.cssText = '';
                    const frameInfo = frameSlider1.querySelector('.frame-info');
                    if (frameInfo) frameInfo.style.cssText = '';
                  }
                  if (modeSwitch1 && controlsBar1.contains(modeSwitch1) && viewport1) {
                    controlsBar1.removeChild(modeSwitch1);
                    viewport1.appendChild(modeSwitch1);
                    modeSwitch1.style.cssText = '';
                    const buttons = modeSwitch1.querySelectorAll('button');
                    buttons.forEach(btn => btn.style.cssText = '');
                  }
                  if (frameInput1 && controlsBar1.contains(frameInput1) && viewport1) {
                    controlsBar1.removeChild(frameInput1);
                    viewport1.appendChild(frameInput1);
                    frameInput1.style.cssText = '';
                    const inputGroup = frameInput1.querySelector('.frame-input-group');
                    if (inputGroup) {
                      inputGroup.style.cssText = '';
                      const label = inputGroup.querySelector('label');
                      if (label) label.style.cssText = '';
                      const input = inputGroup.querySelector('input');
                      if (input) input.style.cssText = '';
                      const goBtn = inputGroup.querySelector('button');
                      if (goBtn) goBtn.style.cssText = '';
                    }
                  }
                }
              }
              
              if (viewport2Wrapper) {
                // Extract viewport2
                if (viewport2 && viewport2Wrapper.contains(viewport2)) {
                  container.appendChild(viewport2);
                }
                // Extract controls from control bar
                const controlsBar2 = viewport2Wrapper.querySelector('div:last-child');
                if (controlsBar2) {
                  if (frameSlider2 && controlsBar2.contains(frameSlider2) && viewport2) {
                    controlsBar2.removeChild(frameSlider2);
                    viewport2.appendChild(frameSlider2);
                    frameSlider2.style.cssText = '';
                    // Reset child styles
                    const sliderLabel = frameSlider2.querySelector('label');
                    if (sliderLabel) sliderLabel.style.cssText = '';
                    const sliderControls = frameSlider2.querySelector('.frame-slider-controls');
                    if (sliderControls) sliderControls.style.cssText = '';
                    const slider = frameSlider2.querySelector('.frame-slider');
                    if (slider) slider.style.cssText = '';
                    const frameInfo = frameSlider2.querySelector('.frame-info');
                    if (frameInfo) frameInfo.style.cssText = '';
                  }
                  if (modeSwitch2 && controlsBar2.contains(modeSwitch2) && viewport2) {
                    controlsBar2.removeChild(modeSwitch2);
                    viewport2.appendChild(modeSwitch2);
                    modeSwitch2.style.cssText = '';
                    const buttons = modeSwitch2.querySelectorAll('button');
                    buttons.forEach(btn => btn.style.cssText = '');
                  }
                  if (frameInput2 && controlsBar2.contains(frameInput2) && viewport2) {
                    controlsBar2.removeChild(frameInput2);
                    viewport2.appendChild(frameInput2);
                    frameInput2.style.cssText = '';
                    const inputGroup = frameInput2.querySelector('.frame-input-group');
                    if (inputGroup) {
                      inputGroup.style.cssText = '';
                      const label = inputGroup.querySelector('label');
                      if (label) label.style.cssText = '';
                      const input = inputGroup.querySelector('input');
                      if (input) input.style.cssText = '';
                      const goBtn = inputGroup.querySelector('button');
                      if (goBtn) goBtn.style.cssText = '';
                    }
                  }
                }
              }
              
              viewportColumn.remove();
            }
            if (treeColumn) {
              if (leftTreeContainer && treeColumn.contains(leftTreeContainer)) viewport1.appendChild(leftTreeContainer);
              if (rightTreeContainer && treeColumn.contains(rightTreeContainer)) viewport2.appendChild(rightTreeContainer);
              treeColumn.remove();
            }

            // Reset container to grid layout
            container.style.display = 'grid';
            container.style.flexDirection = '';
            container.style.gridTemplateColumns = '1fr 1fr';
            container.style.gridTemplateRows = '1fr';

            // Reset viewport styles for side-by-side
            if (viewport1) {
              viewport1.style.cssText = `
                flex: 1;
                width: auto;
                height: auto;
                min-height: auto;
              `;
            }
            if (viewport2) {
              viewport2.style.cssText = `
                flex: 1;
                width: auto;
                height: auto;
                min-height: auto;
              `;
            }

            // Reset tree styles for side-by-side
            if (leftTreeContainer) {
              leftTreeContainer.style.cssText = `
                display: block;
                width: auto;
                height: auto;
                min-height: auto;
                border: 2px solid #e9ecef;
                background: #f8f9fa;
                overflow-y: auto;
                padding: 10px;
                margin-bottom: 10px;
                max-height: 300px;
                box-sizing: border-box;
              `;
            }
            if (rightTreeContainer) {
              rightTreeContainer.style.cssText = `
                display: block;
                width: auto;
                height: auto;
                min-height: auto;
                border: 2px solid #e9ecef;
                background: #f8f9fa;
                overflow-y: auto;
                padding: 10px;
                margin-bottom: 10px;
                max-height: 300px;
                box-sizing: border-box;
              `;
            }
          }
        }

        console.log(
          `S3 viewport layout changed to: ${isStackedLayout ? "stacked" : "side-by-side"}`,
        );
      }

      // Expose toggleViewportLayoutSimple to window for HTML onclick handlers
      window.toggleViewportLayoutSimple = toggleViewportLayoutSimple;

      // Enhanced Progress Management with Cancel Support
      class ProgressManager {
        constructor(viewportNumber) {
          this.viewportNumber = viewportNumber;
          this.currentStep = 0;
          this.totalSteps = 5;
          this.percentage = 0;
          this.metadata = {};
          this.abortController = null;
          this.operationId = null;
          this.startTime = null;
          this.lastProgressUpdate = null;
        }

        show(message = "Loading...", metadata = {}, operationId = null) {
          try {
            const overlay = document.getElementById(
              `progressOverlay${this.viewportNumber}`,
            );
            const cancelButton = document.getElementById(
              `cancelButton${this.viewportNumber}`,
            );

            if (!overlay) {
              console.error(`Progress overlay not found for viewport ${this.viewportNumber}`);
              return;
            }

            // Force visibility with multiple approaches
            overlay.style.display = "flex";
            overlay.style.visibility = "visible";
            overlay.style.opacity = "1";
            overlay.classList.add("active");
            
            // Ensure the overlay is on top
            overlay.style.zIndex = "1000";
            
            this.updateProgress(0);
            this.currentStep = 0;

            // Show cancel button if operation can be cancelled
            if (operationId && cancelButton) {
              this.operationId = operationId;
              cancelButton.style.display = "flex";
            } else if (cancelButton) {
              cancelButton.style.display = "none";
            }

            // Update file and eye information
            this.updateFileInfo(metadata);

            console.log(`Progress shown for viewport ${this.viewportNumber}`);
            console.log(`Overlay display: ${overlay.style.display}, visibility: ${overlay.style.visibility}, opacity: ${overlay.style.opacity}`);
          } catch (error) {
            console.error(`Error showing progress for viewport ${this.viewportNumber}:`, error);
          }
        }

        hide() {
          try {
            const overlay = document.getElementById(
              `progressOverlay${this.viewportNumber}`,
            );
            const cancelButton = document.getElementById(
              `cancelButton${this.viewportNumber}`,
            );

            if (overlay) {
              overlay.classList.remove("active");
              // Add a small delay before hiding to ensure smooth transition
              setTimeout(() => {
                if (overlay && !overlay.classList.contains("active")) {
                  overlay.style.display = "none";
                }
              }, 300);
            }

            if (cancelButton) {
              cancelButton.style.display = "none";
            }

            this.operationId = null;
            this.abortController = null;
            console.log(`Progress hidden for viewport ${this.viewportNumber}`);
          } catch (error) {
            console.error(`Error hiding progress for viewport ${this.viewportNumber}:`, error);
          }
        }

        updateProgress(percentage, message = null) {
          try {
            this.percentage = Math.max(0, Math.min(100, percentage));

            const spinner = document.getElementById(
              `spinner${this.viewportNumber}`,
            );
            const percentageElement = document.getElementById(
              `spinnerPercentage${this.viewportNumber}`,
            );

            // Update spinner color based on progress
            if (spinner) {
              if (this.percentage >= 100) {
                spinner.style.borderTopColor = '#28a745';
              } else if (this.percentage >= 50) {
                spinner.style.borderTopColor = '#338dcc';
              } else {
                spinner.style.borderTopColor = '#338dcc';
              }
            }

            // Update percentage display in spinner center only
            if (percentageElement) {
              percentageElement.textContent = `${Math.round(this.percentage)}%`;
              // Update percentage color to match spinner
              if (this.percentage >= 100) {
                percentageElement.style.color = '#28a745';
              } else if (this.percentage >= 50) {
                percentageElement.style.color = '#338dcc';
              } else {
                percentageElement.style.color = '#338dcc';
              }
            }

            console.log(`Progress updated for viewport ${this.viewportNumber}: ${this.percentage}%`);
          } catch (error) {
            console.error(`Error updating progress for viewport ${this.viewportNumber}:`, error);
          }
        }

        nextStep(message, metadata = {}) {
          this.currentStep = Math.min(this.currentStep + 1, this.totalSteps);
          const stepPercentage = (this.currentStep / this.totalSteps) * 100;
          this.updateProgress(stepPercentage, message);
          this.updateFileInfo(metadata);
        }

        updateMetadata(metadata) {
          // Simplified - no longer used
        }

        updateSteps() {
          // Simplified - no longer used since steps are hidden
        }

        setError(message) {
          this.updateProgress(0, message);
          const spinner = document.getElementById(
            `spinner${this.viewportNumber}`,
          );
          if (spinner) {
            spinner.style.borderTopColor = "#dc3545";
          }
        }

        setAbortController(controller) {
          this.abortController = controller;
        }

        updateEstimatedTime() {
          // Simplified - no longer used
        }

        cancel() {
          if (this.abortController) {
            this.abortController.abort();
            this.updateProgress(0, "Cancelling...");

            setTimeout(() => {
              this.hide();
            }, 1000);
          }
        }
        
        // Method to clean up progress polling
        cleanupProgressPolling() {
          if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
          }
        }

        // Method to update file and eye information
        updateFileInfo(metadata = {}) {
          try {
            const fileNameElement = document.getElementById(`fileName${this.viewportNumber}`);
            const eyeInfoElement = document.getElementById(`eyeInfo${this.viewportNumber}`);

            // Update file name
            if (fileNameElement) {
              if (metadata.fileName) {
                fileNameElement.textContent = metadata.fileName;
              } else if (metadata.File) {
                fileNameElement.textContent = metadata.File;
              } else {
                fileNameElement.textContent = "Loading...";
              }
            }

            // Update eye information
            if (eyeInfoElement) {
              const eyeInfo = [];
              
              if (metadata.eye) {
                eyeInfo.push(`${metadata.eye} Eye`);
              }
              if (metadata.Eye) {
                eyeInfo.push(`${metadata.Eye} Eye`);
              }
              if (metadata.fileType) {
                eyeInfo.push(metadata.fileType.toUpperCase());
              }
              if (metadata.Type) {
                eyeInfo.push(metadata.Type);
              }
              if (metadata.frames) {
                eyeInfo.push(`${metadata.frames} frames`);
              }
              if (metadata.Frames) {
                eyeInfo.push(`${metadata.Frames} frames`);
              }

              if (eyeInfo.length > 0) {
                eyeInfoElement.textContent = eyeInfo.join(" â€¢ ");
              } else {
                eyeInfoElement.textContent = "";
              }
            }

            console.log(`File info updated for viewport ${this.viewportNumber}:`, metadata);
          } catch (error) {
            console.error(`Error updating file info for viewport ${this.viewportNumber}:`, error);
          }
        }
      }

      // Create progress managers for both viewports
      const progressManagers = {
        1: new ProgressManager(1),
        2: new ProgressManager(2),
      };

      // Test function to verify progress system is working
      function testProgressSystem(viewportNumber = 1) {
        console.log(`Testing progress system for viewport ${viewportNumber}`);
        
        if (!progressManagers[viewportNumber]) {
          console.error(`Progress manager not found for viewport ${viewportNumber}`);
          return;
        }

        // Test showing progress
        progressManagers[viewportNumber].show("Testing progress system...", {
          "Test": "Progress",
          "Viewport": viewportNumber
        });

        // Simulate progress updates
        let progress = 0;
        const interval = setInterval(() => {
          progress += 10;
          progressManagers[viewportNumber].updateProgress(progress, `Testing... ${progress}%`);
          
          if (progress >= 100) {
            clearInterval(interval);
            setTimeout(() => {
              progressManagers[viewportNumber].hide();
              console.log("Progress system test completed");
            }, 1000);
          }
        }, 200);
      }

      // Make test function globally available
      window.testProgressSystem = testProgressSystem;

      // Debug function to check progress system elements
      function debugProgressSystem() {
        console.log("=== Progress System Debug ===");
        
        for (let viewport = 1; viewport <= 2; viewport++) {
          console.log(`\nViewport ${viewport}:`);
          
          const elements = {
            overlay: document.getElementById(`progressOverlay${viewport}`),
            text: document.getElementById(`progressText${viewport}`),
            percentage: document.getElementById(`progressPercentage${viewport}`),
            ring: document.getElementById(`progressRing${viewport}`),
            cancelButton: document.getElementById(`cancelButton${viewport}`),
            metadata: document.getElementById(`progressMetadata${viewport}`),
            steps: document.getElementById(`progressSteps${viewport}`)
          };
          
          Object.entries(elements).forEach(([name, element]) => {
            console.log(`  ${name}: ${element ? 'âœ“ Found' : 'âœ— Missing'}`);
          });
          
          console.log(`  Progress Manager: ${progressManagers[viewport] ? 'âœ“ Created' : 'âœ— Missing'}`);
        }
        
        console.log("=== End Debug ===");
      }

      // Make debug function globally available
      window.debugProgressSystem = debugProgressSystem;





      // Cancel operation function
      function cancelOperation(viewportNumber) {
        console.log(`Cancelling operation for viewport ${viewportNumber}`);

        if (progressManagers[viewportNumber]) {
          progressManagers[viewportNumber].cancel();
          progressManagers[viewportNumber].cleanupProgressPolling();
        }


      }

      // Update the existing progress functions
      function showProgress(
        viewportNumber,
        message = "Loading...",
        metadata = {},
      ) {
        console.log(`showProgress called for viewport ${viewportNumber}`);
        console.log(`Progress manager exists: ${!!progressManagers[viewportNumber]}`);
        
        if (!progressManagers[viewportNumber]) {
          console.error(`Progress manager not found for viewport ${viewportNumber}`);
          return null;
        }
        
        const operationId = `operation_${viewportNumber}_${Date.now()}`;
        progressManagers[viewportNumber].show(message, metadata, operationId);
        
        // Verify the progress is actually shown
        setTimeout(() => {
          const overlay = document.getElementById(`progressOverlay${viewportNumber}`);
          if (overlay) {
            console.log(`Progress overlay state after show: display=${overlay.style.display}, classList=${overlay.classList.toString()}`);
          }
        }, 100);
        
        return operationId;
      }

      function hideProgress(viewportNumber) {
        if (progressManagers[viewportNumber]) {
          progressManagers[viewportNumber].cleanupProgressPolling();
          progressManagers[viewportNumber].hide();
        }
      }

      function updateProgress(
        viewportNumber,
        percentage,
        message = null,
        metadata = {},
      ) {
        progressManagers[viewportNumber].updateProgress(percentage, message);
      }

      function nextProgressStep(viewportNumber, message, metadata = {}) {
        progressManagers[viewportNumber].nextStep(message, metadata);
      }

      // Toggle sidebar visibility
      function toggleSidebar() {
        const sidebar = document.querySelector('.sidebar');
        const body = document.body;
        
        if (sidebar.classList.contains('show')) {
          sidebar.classList.remove('show');
          body.classList.remove('sidebar-open');
        } else {
          sidebar.classList.add('show');
          body.classList.add('sidebar-open');
        }
      }

      // Toggle between DICOM view and File Browser
      async function toggleFileFrame() {
        const dicomView = document.getElementById("dicomView");
        const fileBrowser = document.getElementById("fileBrowserContainer");

        if (
          fileBrowser.style.display === "none" ||
          fileBrowser.style.display === ""
        ) {
          const s3Ready = await checkS3Status();
          if (!s3Ready) {
            return;
          }

          dicomView.style.display = "none";
          fileBrowser.style.display = "block";
          if (!s3TreeData) {
            await loadS3Tree();
            // Hide initial loading overlay after S3 tree is loaded
            hideInitialLoadingOverlay();
          } else {
            // If S3 tree data already exists, hide loading overlay immediately
            hideInitialLoadingOverlay();
          }
        } else {
          fileBrowser.style.display = "none";
          dicomView.style.display = "flex";
        }
      }

      // Enhanced Tree Browser with Drill-down Navigation and Sorting
      class S3TreeBrowser {
        constructor() {
          this.currentPath = [];
          this.currentData = null;
          this.rootData = null;
          this.selectedItem = null;
          this.isLoading = false;
        }

        async initialize() {
          await this.loadRootLevel();
        }

        async loadRootLevel() {
          try {
            this.showLoading();
            console.log("Loading S3 root tree with lazy loading...");

            // Use lazy loading instead of loading all files at once
            const response = await fetch("/api/s3-lazy-tree?path=&max_items=100");

            if (response.status === 503) {
              const data = await response.json();
              if (data.needs_credentials) {
                showS3CredentialsModal();
                return;
              }
            }

            if (!response.ok) {
              throw new Error(
                `HTTP ${response.status}: ${response.statusText}`,
              );
            }

            const data = await response.json();
            console.log("S3 lazy tree data received:", data);

            // Convert lazy tree data to tree structure
            this.rootData = this.convertLazyTreeToTreeStructure(data);
            this.currentData = this.rootData;
            this.currentPath = [];
            this.renderCurrentLevel();
            this.hideLoading();

            // Optional: hide the progress display
            document
              .getElementById("treeProgressBarContainer")
              ?.style?.setProperty("display", "none", "important");
          } catch (error) {
            console.error("Error loading S3 list:", error);
            this.showError(`Failed to load S3 tree: ${error.message}`);
          }
        }

        // New method to convert lazy tree data to tree structure
        convertLazyTreeToTreeStructure(lazyData) {
          const root = [];
          
          // Add folders
          if (lazyData.folders) {
            lazyData.folders.forEach(folder => {
              root.push({
                name: folder.name,
                type: 'folder',
                path: folder.path,
                has_children: folder.has_children,
                children: [] // Will be loaded lazily
              });
            });
          }
          
          // Add files (include all files, mark unsupported ones)
          if (lazyData.files) {
            lazyData.files.forEach(file => {
              root.push({
                name: file.name,
                type: 'file',
                path: file.path,
                size: file.size,
                last_modified: file.last_modified,
                unsupported: this.isUnsupportedFileType(file.name) // Mark as unsupported
              });
            });
          }
          
          // Apply custom sorting
          return sortTreeStructure(root);
        }
        
        // Check if file type is unsupported
        isUnsupportedFileType(filename) {
          const lowerName = filename.toLowerCase();
          return lowerName.endsWith('.ex.dcm') || 
                 lowerName.match(/\.e\d{3}\.ecm$/i) ||
                 lowerName.endsWith('.ex.dicom') ||
                 lowerName.endsWith('.ex.dcm') ||
                 lowerName.endsWith('.ecm');
        }

        // New method for lazy loading folder contents
        async loadFolderContents(folderPath) {
          try {
            console.log(`Lazy loading folder: ${folderPath}`);
            
            const response = await fetch(`/api/s3-lazy-tree?path=${encodeURIComponent(folderPath)}&max_items=100`);
            
            if (!response.ok) {
              throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return this.convertLazyTreeToTreeStructure(data);
          } catch (error) {
            console.error(`Error loading folder ${folderPath}:`, error);
            return [];
          }
        }

        buildTreeFromFlatList(files) {
          const root = [];

          files.forEach((file) => {
            const parts = file.key.split("/");
            let current = root;

            parts.forEach((part, index) => {
              let node = current.find((n) => n.name === part);
              if (!node) {
                node = {
                  name: part,
                  type: index === parts.length - 1 ? "file" : "folder",
                  ...(index === parts.length - 1
                    ? {
                        size: file.size,
                        last_modified: file.last_modified,
                        path: file.key,
                      }
                    : { children: [] }),
                };
                current.push(node);
              }
              if (node.type === "folder") {
                current = node.children;
              }
            });
          });

          // Apply custom sorting to the entire tree
          return sortTreeStructure(root);
        }

        showLoading() {
          const container = document.getElementById("fileTreeContainer");
          container.innerHTML = `
      <div class="tree-loading">
        <div class="loader"></div>
        <span style="margin-left: 10px;">Loading...</span>
      </div>
    `;
          this.isLoading = true;
        }

        hideLoading() {
          this.isLoading = false;
        }

        showError(message) {
          const container = document.getElementById("fileTreeContainer");
          container.innerHTML = `
      <div class="tree-error">
        <p>${message}</p>
        <button class="retry-button" onclick="s3Browser.loadRootLevel()">Retry</button>
      </div>
    `;
          this.hideLoading();
        }

        renderCurrentLevel() {
          this.updateNavigation();
          this.renderItems();
        }

        renderSearchResults(results) {
          const container = document.getElementById("fileTreeContainer");

          if (!results.length) {
            container.innerHTML =
              '<div class="tree-empty">No matching files or folders found.</div>';
            return;
          }

          console.log("Rendering search results:", results);

          // Apply sorting to search results
          const sortedResults = sortTreeStructure([...results]);

          const treeList = document.createElement("ul");
          treeList.className = "file-tree";

          sortedResults.forEach((result) => {
            // Create a proper tree item using the existing createTreeItem method
            // We need to reconstruct the item structure that createTreeItem expects
            const item = {
              name: result.name,
              type: result.type,
              path: result.path || result.fullPath, // Handle both path formats
              size: result.size,
              unsupported: result.unsupported,
              children: result.children || []
            };

            console.log("Creating tree item for:", item);

            // Use the existing createTreeItem method to ensure consistency
            const listItem = this.createTreeItem(item, []);
            
            // Highlight search term in the name if there's a search query
            const searchInput = document.getElementById("treeSearchInput");
            const query = searchInput.value.toLowerCase().trim();
            if (query && result.name) {
              const nameElement = listItem.querySelector('.tree-item-name');
              if (nameElement) {
                const highlightedName = result.name.replace(
                  new RegExp(`(${query})`, 'gi'),
                  '<mark style="background-color: #ffeb3b; padding: 1px 2px; border-radius: 2px;">$1</mark>'
                );
                nameElement.innerHTML = highlightedName;
              }
            }
            
            treeList.appendChild(listItem);
          });

          container.innerHTML = "";
          container.appendChild(treeList);

          // Re-apply selection mode if it was active
          if (isSelectionMode) {
            setTimeout(() => {
              addSelectionCheckboxes();
            }, 50);
          }
        }

        updateNavigation() {
          const backButton = document.getElementById("backButton");
          const currentPathSpan = document.getElementById("currentPath");

          if (this.currentPath.length > 0) {
            backButton.classList.add("visible");
            currentPathSpan.textContent = this.currentPath.join(" / ");
          } else {
            backButton.classList.remove("visible");
            currentPathSpan.textContent = "Root";
          }
        }

        renderItems() {
          const container = document.getElementById("fileTreeContainer");

          if (!this.currentData || this.currentData.length === 0) {
            container.innerHTML =
              '<div class="tree-empty">No files or folders found</div>';
            return;
          }

          // Apply sorting to current level before rendering
          const sortedData = sortTreeStructure([...this.currentData]);

          const treeList = document.createElement("ul");
          treeList.className = "file-tree";

          sortedData.forEach((item) => {
            const listItem = this.createTreeItem(item, this.currentPath);
            treeList.appendChild(listItem);
          });

          container.innerHTML = "";
          container.appendChild(treeList);

          // Re-apply selection mode if it was active
          if (isSelectionMode) {
            setTimeout(() => {
              addSelectionCheckboxes();
            }, 50);
          }
        }

        // ...inside S3TreeBrowser class...

        createTreeItem(item, parentPath = []) {
          const li = document.createElement("li");

          const treeItem = document.createElement("div");
          treeItem.className = "tree-item";

          // Build full path for this item
          const fullPath = [...parentPath, item.name].join("/");

          // Always set data-full-path for both files and folders
          treeItem.dataset.fullPath = fullPath;
          treeItem.dataset.name = item.name;
          treeItem.dataset.type = item.type;

          // Add appropriate class for ALL items to enable selection
          if (item.type === "file") {
            treeItem.classList.add("file-item");
            treeItem.dataset.filePath = item.path || fullPath;
            
            // Mark unsupported files
            if (item.unsupported) {
              treeItem.classList.add("unsupported-file");
              treeItem.dataset.unsupported = "true";
            }
          } else if (item.type === "folder") {
            treeItem.classList.add("folder-item");
          }

          // === Click behavior (select or expand) ===
          treeItem.onclick = (e) => {
            if (isSelectionMode) {
              handleFileSelection(e);
            } else {
              // Handle unsupported files - show notification and prevent loading
              if (item.type === "file" && item.unsupported) {
                e.preventDefault();
                e.stopPropagation();
                
                // Set selected file path for visual feedback
                selectedFilePath = item.path;
                this.selectedItem = item;
                
                // Clear previous selection and select this item
                document.querySelectorAll(".tree-item.selected").forEach((el) => {
                  el.classList.remove("selected");
                });
                treeItem.classList.add("selected");
                
                // Show notification about work in progress file type
                showNotification("This file type is work in progress and cannot be loaded yet.", "error");
                
                return;
              } else {
                this.handleItemClick(item, treeItem);
              }
            }
          };

          // === Right-click context menu for FILES only ===
          if (item.type === "file" && item.path) {
            treeItem.addEventListener("contextmenu", (e) => {
              e.preventDefault();
              
              // Prevent context menu for work in progress files
              if (item.unsupported) {
                showNotification("This file type is work in progress and cannot be loaded yet.", "error");
                return;
              }
              
              selectedFilePath = item.path;
              this.selectedItem = item;

              // Clear previous selection and select this item
              document.querySelectorAll(".tree-item.selected").forEach((el) => {
                el.classList.remove("selected");
              });
              treeItem.classList.add("selected");

              showContextMenu(e, item.path);
            });
          }

          // === Content container ===
          const content = document.createElement("div");
          content.className = "tree-item-content";

          // === Expand arrow (only folders) ===
          const arrow = document.createElement("div");
          arrow.className = "expand-arrow collapsed";
          arrow.textContent = item.type === "folder" ? ">" : "";
          if (item.type !== "folder") {
            arrow.style.width = "16px"; // spacer for file rows
          }
          content.appendChild(arrow);

          // === Icon ===
          const icon = document.createElement("span");
          icon.className = "tree-icon";
          icon.innerHTML =
            item.type === "folder"
              ? '<i class="fas fa-folder folder-icon"></i>'
              : '<i class="fas fa-file-medical file-icon"></i>';
          content.appendChild(icon);

          // === Name ===
          const name = document.createElement("span");
          name.className = "tree-item-name";
          name.textContent = item.name || "Unnamed";
          content.appendChild(name);

          // === Work in Progress Tag ===
          if (item.type === "file" && item.unsupported) {
            const workInProgressTag = document.createElement("span");
            workInProgressTag.className = "work-in-progress-tag";
            workInProgressTag.innerHTML = '<i class="fas fa-clock"></i> Work in Progress';
            content.appendChild(workInProgressTag);
          }

          // === File size (for files only) ===
          if (item.type === "file" && item.size) {
            const info = document.createElement("span");
            info.className = "tree-item-info";
            info.textContent = this.formatFileSize(item.size);
            content.appendChild(info);
          }

          // === Selection checkbox (for ALL items - files and folders) ===
          const checkbox = document.createElement("div");
          checkbox.className = "selection-checkbox";
          content.appendChild(checkbox);

          // === Tooltip with full filename ===
          const tooltip = document.createElement("div");
          tooltip.className = "tree-item-tooltip";
          tooltip.textContent = item.name || "";

          // === Assemble final node ===
          treeItem.appendChild(content);
          treeItem.appendChild(tooltip);
          li.appendChild(treeItem);

          // If folder, recursively add children (not shown here, handled in renderItems)
          return li;
        }

        async handleItemClick(item, element) {
          if (this.isLoading) return;

          // If in selection mode, don't handle normal item clicking
          if (isSelectionMode) {
            return;
          }

          // Clear previous selection (only in normal mode)
          document.querySelectorAll(".tree-item.selected").forEach((el) => {
            el.classList.remove("selected");
          });

          // Select current item
          element.classList.add("selected");
          this.selectedItem = item;

          if (item.type === "folder") {
            await this.drillIntoFolder(item);
          } else {
            console.log("File selected:", item.path);
            selectedFilePath = item.path;
          }
        }

        async drillIntoFolder(folder) {
          try {
            // Show loading state
            this.showLoading();

            // Update path
            this.currentPath.push(folder.name);

            // Load folder contents using lazy loading
            let folderContents;
            if (folder.children && folder.children.length > 0) {
              // Use cached children if available
              folderContents = folder.children;
            } else {
              // Load folder contents lazily from API
              const folderPath = folder.path;
              folderContents = await this.loadFolderContents(folderPath);
              
              // Cache the loaded children
              folder.children = folderContents;
            }

            this.currentData = folderContents;
            this.renderCurrentLevel();
            this.hideLoading();

            console.log(`Drilled into folder: ${this.currentPath.join("/")}`);
          } catch (error) {
            console.error("Error drilling into folder:", error);
            this.showError(`Failed to load folder contents: ${error.message}`);
            // Revert path on error
            this.currentPath.pop();
          }
        }

        goBack() {
          if (this.currentPath.length === 0) return;

          this.currentPath.pop();

          if (this.currentPath.length === 0) {
            // Back to root
            this.currentData = this.rootData;
          } else {
            // Navigate to parent folder
            this.currentData = this.findFolderByPath(this.currentPath);
          }

          this.renderCurrentLevel();
          console.log(
            `Navigated back to: ${this.currentPath.join("/") || "root"}`,
          );
        }

        findFolderByPath(path) {
          let current = this.rootData;

          for (const segment of path) {
            const folder = current.find(
              (item) => item.name === segment && item.type === "folder",
            );
            if (folder && folder.children) {
              current = folder.children;
            } else {
              console.warn("Could not find path:", path);
              return this.rootData;
            }
          }

          return current;
        }

        handleFileRightClick(e, item, element) {
          e.preventDefault();
          e.stopPropagation();

          console.log("Right-click detected on file:", item.path);
          selectedFilePath = item.path;
          this.selectedItem = item;

          // Clear previous selection and select this item
          document.querySelectorAll(".tree-item.selected").forEach((el) => {
            el.classList.remove("selected");
          });
          element.classList.add("selected");

          showContextMenu(e, item);
        }

        formatFileSize(bytes) {
          if (bytes === 0) return "0 B";
          const k = 1024;
          const sizes = ["B", "KB", "MB", "GB"];
          const i = Math.floor(Math.log(bytes) / Math.log(k));
          return (
            parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i]
          );
        }

        // Search functionality
        async search(query) {
          if (!query.trim()) {
            this.renderCurrentLevel();
            return;
          }

          try {
            this.showLoading();
            
            // Use the new search endpoint with lazy loading
            const response = await fetch(`/api/s3-search?query=${encodeURIComponent(query)}&max_results=50`);
            
            if (!response.ok) {
              throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log("Search results:", data);

            const container = document.getElementById("fileTreeContainer");

            if (data.results.length === 0) {
              container.innerHTML = '<div class="tree-empty">No matching files found</div>';
              this.hideLoading();
              return;
            }

            // Convert search results to tree items (include all files, mark unsupported ones)
            const searchItems = data.results.map(result => ({
              name: result.name,
              type: 'file',
              path: result.key,
              size: result.size,
              last_modified: result.last_modified,
              unsupported: this.isUnsupportedFileType(result.name) // Mark as unsupported
            }));

                        // Apply sorting to filtered results
            const sortedFilteredItems = sortTreeStructure([...searchItems]);

            const treeList = document.createElement("ul");
            treeList.className = "file-tree";

            sortedFilteredItems.forEach((item) => {
              const listItem = this.createTreeItem(item, []);
              treeList.appendChild(listItem);
            });

            container.innerHTML = "";
            container.appendChild(treeList);
            this.hideLoading();
          } catch (error) {
            console.error("Search error:", error);
            this.showError(`Search failed: ${error.message}`);
          }
        }
      }

      // Global tree search
      function searchS3Tree(query, tree, path = []) {
        const results = [];

        if (!tree || !Array.isArray(tree)) {
          console.warn("Invalid tree data for search:", tree);
          return results;
        }

        tree.forEach((node) => {
          if (!node || !node.name) {
            console.warn("Invalid node in tree:", node);
            return;
          }

          const fullPath = [...path, node.name];
          if (node.name.toLowerCase().includes(query)) {
            results.push({ ...node, fullPath: fullPath.join("/") });
          }

          if (node.type === "folder" && node.children?.length > 0) {
            results.push(...searchS3Tree(query, node.children, fullPath));
          }
        });

        console.log(`Search for "${query}" returned ${results.length} results`);
        return results;
      }

      // Initialize the browser
      let s3Browser = new S3TreeBrowser();
      window.s3Browser = s3Browser;

      // Selection mode variables
      let isSelectionMode = false;
      let selectedFiles = new Set();
      let cachedFiles = new Map();

      // Load cached files from localStorage
      try {
        const stored = localStorage.getItem('s3CachedFiles');
        if (stored) {
          cachedFiles = new Map(JSON.parse(stored));
        }
      } catch (e) {
        console.error('Error loading cached files:', e);
      }

      function toggleSelectionMode() {
        isSelectionMode = !isSelectionMode;
        selectedFiles.clear();
        
        const selectBtn = document.getElementById('selectModeBtn');
        const cancelBtn = document.getElementById('cancelSelection');
        
        if (isSelectionMode) {
          selectBtn.innerHTML = '<i class="fas fa-check-square"></i> Selecting...';
          selectBtn.style.backgroundColor = '#ffc107';
          selectBtn.style.color = '#000';
          cancelBtn.style.display = 'flex';
          
          // Add selection capability to all items
          addSelectionCheckboxes();
          console.log('Selection mode enabled');
        } else {
          cancelSelectionMode();
        }
      }

      function cancelSelectionMode() {
        isSelectionMode = false;
        selectedFiles.clear();
        
        const selectBtn = document.getElementById('selectModeBtn');
        const cancelBtn = document.getElementById('cancelSelection');
        
        selectBtn.innerHTML = '<i class="fas fa-check-square"></i> Select';
        selectBtn.style.backgroundColor = '#007bff';
        selectBtn.style.color = 'white';
        cancelBtn.style.display = 'none';
        
        // Remove selection styles and checkboxes
        removeSelectionCheckboxes();
        console.log('Selection mode cancelled');
      }

      function addSelectionCheckboxes() {
        // Add selection capability to ALL tree items (files AND folders)
        const allItems = document.querySelectorAll('.tree-item');
        allItems.forEach(item => {
          item.classList.add('file-item-selectable');

          // Show existing checkbox or create one
          let checkbox = item.querySelector('.selection-checkbox');
          if (!checkbox) {
            checkbox = document.createElement('div');
            checkbox.className = 'selection-checkbox';
            const content = item.querySelector('.tree-item-content');
            if (content) {
              content.appendChild(checkbox);
            }
          }
          checkbox.classList.add('visible');

          // Check if this is an E2E file and disable it
          const itemPath = item.dataset.fullPath;
          if (itemPath && itemPath.toLowerCase().endsWith('.e2e')) {
            checkbox.classList.add('disabled');
            checkbox.title = 'E2E files are cached when loaded, not pre-cached';
            checkbox.style.opacity = '0.5';
            checkbox.style.cursor = 'not-allowed';
          } else {
            checkbox.classList.remove('disabled');
            checkbox.title = '';
            checkbox.style.opacity = '1';
            checkbox.style.cursor = 'pointer';
          }

          // Remove any previous handler to avoid duplicates
          checkbox.onclick = null;
          checkbox.addEventListener('click', handleFileSelection);

          // Prevent the row itself from handling selection in selection mode
          // (let it only handle navigation/expand)
          // Remove any previous click handler for selection mode
          if (item._selectionHandlerAdded) {
            item.removeEventListener('click', handleFileSelection);
            item._selectionHandlerAdded = false;
          }
        });
      }

      function removeSelectionCheckboxes() {
        const allItems = document.querySelectorAll('.file-item-selectable');
        allItems.forEach(item => {
          item.classList.remove('file-item-selectable', 'selected');
          const checkbox = item.querySelector('.selection-checkbox');
          if (checkbox) {
            checkbox.classList.remove('visible', 'checked');
          }
          // Remove only the selection handler
          if (item._selectionHandlerAdded) {
            item.removeEventListener('click', handleFileSelection);
            item._selectionHandlerAdded = false;
          }
        });
      }
      //file handelling-- for s3 bucket 

      function handleFileSelection(event) {
        if (!isSelectionMode) return;

        event.stopPropagation();
        event.preventDefault();

        // The checkbox is the event target, get its parent .tree-item
        const checkbox = event.currentTarget;
        
        // Prevent selection of disabled E2E files
        if (checkbox.classList.contains('disabled')) {
          return;
        }
        
        const treeItem = checkbox.closest('.tree-item');
        const itemPath = treeItem.dataset.fullPath;

        if (!itemPath) {
          console.warn('No full path found for item');
          return;
        }

        // Prevent selection of work in progress files
        if (treeItem.classList.contains('unsupported-file')) {
          showNotification("This file type is work in progress and cannot be selected.", "error");
          return;
        }

        if (selectedFiles.has(itemPath)) {
          selectedFiles.delete(itemPath);
          treeItem.classList.remove('selected');
          checkbox.classList.remove('checked');
        } else {
          selectedFiles.add(itemPath);
          treeItem.classList.add('selected');
          checkbox.classList.add('checked');
        }

        console.log(`Selected items: ${selectedFiles.size}`, [...selectedFiles]);
      }

      // Handle real-time progress updates from SSE
      function handleCacheProgressUpdate(data) {
        console.log('Progress update:', data);
        
        switch (data.type) {
          case 'progress':
            updateOverallProgress(data.percentage, data.completed, data.total);
            break;
          case 'file_start':
            updateCurrentFile(data.file, data.index, data.total);
            break;
          case 'download_start':
            updateDownloadStatus(data.file, data.message);
            break;

          case 'download_complete':
            updateDownloadComplete(data.file, data.message);
            break;
          case 'processing_start':
            updateProcessingStatus(data.file, data.message);
            break;
          case 'crc_calculated':
            updateCRCStatus(data.file, data.crc);
            break;
          case 'processing':
            updateProcessingStatus(data.file, data.message);
            break;
          case 'file_complete':
            updateFileComplete(data.file, data.status, data.crc);
            if (data.crc) {
              fileCRCs[data.file] = data.crc;
            }
            break;
          case 'file_error':
            updateFileError(data.file, data.error);
            // Show backend error message in a notification for the user
            showNotification(`Error caching file: ${data.file}\n${data.error}`, 'error');
            break;
          case 'skipped':
            updateFileSkipped(data.file, data.message);
            break;
          case 'complete':
            handleCachingComplete(data.total, data.completed);
            break;
        }
      }

      // Progress popup functions with enhanced real-time updates
      function showCacheProgressPopup(totalFiles, streaming = false) {
        // Remove existing popup if any
        const existingPopup = document.getElementById('cacheProgressPopup');
        if (existingPopup) {
          existingPopup.remove();
        }

        const popup = document.createElement('div');
        popup.id = 'cacheProgressPopup';
        popup.innerHTML = `
          <div class="cache-progress-overlay">
            <div class="cache-progress-modal">
              <div class="cache-progress-header">
                <h3><i class="fas fa-cloud-download-alt"></i> Caching Files</h3>
                <button class="cache-progress-close" onclick="hideCacheProgressPopup()">Ã—</button>
              </div>
              <div class="cache-progress-content">
                <div class="cache-progress-bar">
                  <div class="cache-progress-fill" id="cacheProgressFill"></div>
                </div>
                <div class="cache-progress-text" id="cacheProgressText">Preparing to cache ${totalFiles} files...</div>
                
                <!-- Current file progress -->
                <div class="current-file-section" id="currentFileSection" style="display: none;">
                  <div class="current-file-header">
                    <h4 id="currentFileName">Current File</h4>
                    <span id="currentFileStatus" class="file-status">Preparing...</span>
                  </div>
                  <div class="current-file-progress-bar">
                    <div class="current-file-progress-fill" id="currentFileProgressFill"></div>
                  </div>
                  <div class="current-file-details" id="currentFileDetails"></div>
                </div>
                
                <div class="cache-progress-stats" id="cacheProgressStats">
                  <div class="stat-item">
                    <span class="stat-label">Total:</span>
                    <span class="stat-value">${totalFiles}</span>
                  </div>
                  <div class="stat-item">
                    <span class="stat-label">Cached:</span>
                    <span class="stat-value" id="cachedCount">0</span>
                  </div>
                  <div class="stat-item">
                    <span class="stat-label">Skipped:</span>
                    <span class="stat-value" id="skippedCount">0</span>
                  </div>
                  <div class="stat-item">
                    <span class="stat-label">Failed:</span>
                    <span class="stat-value" id="failedCount">0</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        `;
        document.body.appendChild(popup);
      }

      function updateOverallProgress(percentage, completed, total) {
        const progressFill = document.getElementById('cacheProgressFill');
        const progressText = document.getElementById('cacheProgressText');
        
        if (progressFill) {
          progressFill.style.width = `${percentage}%`;
        }
        
        if (progressText) {
          progressText.textContent = `Overall Progress: ${Math.round(percentage)}% (${completed}/${total})`;
        }
      }

      function updateCurrentFile(file, index, total) {
        const currentFileSection = document.getElementById('currentFileSection');
        const currentFileName = document.getElementById('currentFileName');
        const currentFileStatus = document.getElementById('currentFileStatus');
        
        if (currentFileSection) {
          currentFileSection.style.display = 'block';
        }
        
        if (currentFileName) {
          const fileName = file.split('/').pop();
          currentFileName.textContent = `File ${index}/${total}: ${fileName}`;
        }
        
        if (currentFileStatus) {
          currentFileStatus.textContent = 'Starting...';
          currentFileStatus.className = 'file-status preparing';
        }
      }

      function updateDownloadStatus(file, message) {
        const currentFileStatus = document.getElementById('currentFileStatus');
        const currentFileDetails = document.getElementById('currentFileDetails');
        
        if (currentFileStatus) {
          currentFileStatus.textContent = 'Downloading...';
          currentFileStatus.className = 'file-status downloading';
        }
        
        if (currentFileDetails) {
          currentFileDetails.textContent = message;
        }
      }

      function updateProcessingStatus(file, message) {
        const currentFileStatus = document.getElementById('currentFileStatus');
        const currentFileDetails = document.getElementById('currentFileDetails');
        
        if (currentFileStatus) {
          currentFileStatus.textContent = 'Processing...';
          currentFileStatus.className = 'file-status processing';
        }
        
        if (currentFileDetails) {
          currentFileDetails.textContent = message;
        }
      }



      function updateDownloadComplete(file, message) {
        const currentFileStatus = document.getElementById('currentFileStatus');
        const currentFileDetails = document.getElementById('currentFileDetails');
        
        if (currentFileStatus) {
          currentFileStatus.textContent = 'Processing...';
          currentFileStatus.className = 'file-status processing';
        }
        
        if (currentFileDetails) {
          currentFileDetails.textContent = message;
        }
      }

      function updateProcessingStatus(file, message) {
        const currentFileStatus = document.getElementById('currentFileStatus');
        const currentFileDetails = document.getElementById('currentFileDetails');
        
        if (currentFileStatus) {
          currentFileStatus.textContent = 'Processing...';
          currentFileStatus.className = 'file-status processing';
        }
        
        if (currentFileDetails) {
          currentFileDetails.textContent = message;
        }
      }

      function updateCRCStatus(file, crc) {
        const currentFileDetails = document.getElementById('currentFileDetails');
        
        if (currentFileDetails) {
          currentFileDetails.textContent = `CRC calculated: ${crc}`;
        }
      }

      function updateFileComplete(file, status, crc) {
        const currentFileStatus = document.getElementById('currentFileStatus');
        const currentFileDetails = document.getElementById('currentFileDetails');
        const cachedCount = document.getElementById('cachedCount');
        
        if (currentFileStatus) {
          currentFileStatus.textContent = 'Complete';
          currentFileStatus.className = 'file-status complete';
        }
        
        if (currentFileDetails) {
          currentFileDetails.textContent = `Successfully cached (CRC: ${crc})`;
        }
        
        if (cachedCount) {
          const current = parseInt(cachedCount.textContent) || 0;
          cachedCount.textContent = current + 1;
        }
      }

      function updateFileError(file, error) {
        const currentFileStatus = document.getElementById('currentFileStatus');
        const currentFileDetails = document.getElementById('currentFileDetails');
        const failedCount = document.getElementById('failedCount');
        
        if (currentFileStatus) {
          currentFileStatus.textContent = 'Failed';
          currentFileStatus.className = 'file-status failed';
        }
        
        if (currentFileDetails) {
          currentFileDetails.textContent = `Error: ${error}`;
        }
        
        if (failedCount) {
          const current = parseInt(failedCount.textContent) || 0;
          failedCount.textContent = current + 1;
        }
      }

      function updateFileSkipped(file, message) {
        const currentFileStatus = document.getElementById('currentFileStatus');
        const currentFileDetails = document.getElementById('currentFileDetails');
        const skippedCount = document.getElementById('skippedCount');
        
        if (currentFileStatus) {
          currentFileStatus.textContent = 'Skipped';
          currentFileStatus.className = 'file-status skipped';
        }
        
        if (currentFileDetails) {
          currentFileDetails.textContent = message;
        }
        
        if (skippedCount) {
          const current = parseInt(skippedCount.textContent) || 0;
          skippedCount.textContent = current + 1;
        }
      }

      function handleCachingComplete(total, completed) {
        const progressText = document.getElementById('cacheProgressText');
        const currentFileSection = document.getElementById('currentFileSection');
        
        if (progressText) {
          progressText.textContent = 'Caching complete!';
        }
        
        if (currentFileSection) {
          currentFileSection.style.display = 'none';
        }
        
        // Update save button
          const saveBtn = document.getElementById('saveToCache');
        if (saveBtn) {
          saveBtn.innerHTML = '<i class="fas fa-check"></i> Saved!';
          saveBtn.style.backgroundColor = '#28a745';
          setTimeout(() => {
            saveBtn.innerHTML = `<i class="fas fa-save"></i> Save to Cache`;
            saveBtn.style.backgroundColor = '#28a745';
          }, 2000);
        }

        // Clear selection
          selectedFiles.clear();
          removeSelectionCheckboxes();
          addSelectionCheckboxes();
        if (saveBtn) {
          saveBtn.style.display = 'none';
        }
        
        // Auto-hide popup after completion (stay on current page)
        setTimeout(() => {
          hideCacheProgressPopup();
        }, 3000);
      }
      
      // Function to navigate to viewport and load the first cached file
      async function navigateToViewportAndLoadFirstFile() {
        try {
          // Get the first cached file from the fileCRCs object
          const cachedFiles = Object.keys(fileCRCs);
          if (cachedFiles.length === 0) {
            // Just show viewport mode without loading any file
            showViewportMode();
            return;
          }
          
          const firstCachedFile = cachedFiles[0];
          console.log(`Navigating to viewport and loading first cached file: ${firstCachedFile}`);
          
          // Show viewport mode
          showViewportMode();
          
          // Load the first cached file into viewport 1
          selectedFilePath = firstCachedFile;
          await loadIntoViewport(1);
          
          showNotification(`Loaded ${firstCachedFile.split('/').pop()} into viewport`, 'success');
          
        } catch (error) {
          console.error('Error navigating to viewport:', error);
          showNotification(`Error loading cached file: ${error.message}`, 'error');
        }
      }
      
      // Function to show viewport mode
      function showViewportMode() {
        // Hide file browser
        const fileBrowserContainer = document.querySelector('.file-browser-container');
        if (fileBrowserContainer) {
          fileBrowserContainer.style.display = 'none';
        }
        
        // Show viewports container
        const viewportsContainer = document.getElementById('viewportsContainer');
        if (viewportsContainer) {
          viewportsContainer.style.display = 'flex';
          viewportsContainer.style.flexDirection = 'row';
          viewportsContainer.style.gap = '20px';
        }
        
        // Show both viewports
        const viewport1 = document.getElementById('viewport1');
        const viewport2 = document.getElementById('viewport2');
        
        if (viewport1) {
          viewport1.style.display = 'flex';
          viewport1.style.border = '2px solid #ccc';
          viewport1.style.width = '';
          viewport1.style.flex = '1';
        }
        
        if (viewport2) {
          viewport2.style.display = 'flex';
          viewport2.style.border = '2px solid #ccc';
          viewport2.style.width = '';
          viewport2.style.flex = '1';
        }
        
        console.log('Switched to viewport mode');
      }

      function hideCacheProgressPopup() {
        const popup = document.getElementById('cacheProgressPopup');
        if (popup) {
          popup.remove();
        }
      }

      // Notification system to replace alerts
      function showNotification(message, type = 'info', duration = 5000) {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(notification => notification.remove());

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
          <div class="notification-content">
            <span class="notification-message">${message}</span>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">Ã—</button>
          </div>
        `;

        // Add styles if not already present
        if (!document.getElementById('notification-styles')) {
          const style = document.createElement('style');
          style.id = 'notification-styles';
          style.textContent = `
            .notification {
              position: fixed;
              top: 20px;
              right: 20px;
              z-index: 10000;
              max-width: 400px;
              border-radius: 8px;
              box-shadow: 0 4px 12px rgba(0,0,0,0.15);
              animation: slideIn 0.3s ease-out;
            }
            .notification-info {
              background: #e3f2fd;
              border: 1px solid #2196f3;
              color: #1976d2;
            }
            .notification-warning {
              background: #fff3e0;
              border: 1px solid #ff9800;
              color: #f57c00;
            }
            .notification-error {
              background: #ffebee;
              border: 1px solid #f44336;
              color: #d32f2f;
            }
            .notification-success {
              background: #e8f5e8;
              border: 1px solid #4caf50;
              color: #388e3c;
            }
            .notification-content {
              padding: 12px 16px;
              display: flex;
              align-items: center;
              justify-content: space-between;
            }
            .notification-message {
              flex: 1;
              margin-right: 8px;
            }
            .notification-close {
              background: none;
              border: none;
              font-size: 18px;
              cursor: pointer;
              color: inherit;
              opacity: 0.7;
            }
            .notification-close:hover {
              opacity: 1;
            }
            @keyframes slideIn {
              from { transform: translateX(100%); opacity: 0; }
              to { transform: translateX(0); opacity: 1; }
            }
          `;
          document.head.appendChild(style);
        }

        document.body.appendChild(notification);

        // Auto-remove after duration
        setTimeout(() => {
          if (notification.parentElement) {
            notification.remove();
          }
        }, duration);
      }

      function reviewCachedFiles() {
        const cached = [...cachedFiles.entries()];
        console.log('Cached items:', cached);
        
        // Display cached items in a more user-friendly way
        const cacheDisplay = cached.map(([path, data]) => ({
          path: path,
          type: data.type,
          cachedAt: new Date(data.cachedAt).toLocaleString()
        }));
        
        console.table(cacheDisplay);
        return cached;
      }

      // Show selection controls when S3 browser is loaded
      document.addEventListener('DOMContentLoaded', () => {
        const controls = document.querySelector('.s3-selection-controls');
        if (controls) {
          controls.style.display = 'flex';
          document.getElementById('selectModeBtn').style.display = 'block';
        }
      });

      // Update search event listener
      function runTreeSearch() {
        const query = document
          .getElementById("treeSearchInput")
          .value.toLowerCase()
          .trim();
        const extension = document.getElementById("extensionFilter").value;

        console.log("Running tree search:", { query, extension });

        // Update search status indicator
        const statusIndicator = document.getElementById("s3StatusIndicator");
        if (query || extension) {
          statusIndicator.innerHTML = `
            <span class="status-dot checking"></span>
            <span class="status-text checking">Searching...</span>
            <button class="status-refresh-btn" onclick="refreshS3Status()" title="Refresh connection">
              <i class="fas fa-sync-alt"></i>
            </button>
          `;
        }

        if (!query && !extension) {
          console.log("No search criteria, showing default tree");
          s3Browser.renderCurrentLevel(); // Show default tree
          // Restore normal status
          checkS3Status();
          return;
        }

        if (!s3Browser.rootData) {
          console.warn("No root data available for search");
          return;
        }

        let results = searchS3Tree(query, s3Browser.rootData || []);

        if (extension) {
          const beforeFilter = results.length;
          results = results.filter(
            (item) =>
              item.type === "file" &&
              item.name.toLowerCase().endsWith(extension),
          );
          console.log(`Extension filter "${extension}" reduced results from ${beforeFilter} to ${results.length}`);
        }

        console.log("Final search results:", results);
        s3Browser.renderSearchResults(results);

        // Update status to show search results
        statusIndicator.innerHTML = `
          <span class="status-dot connected"></span>
          <span class="status-text connected">Found ${results.length} results</span>
          <button class="status-refresh-btn" onclick="refreshS3Status()" title="Refresh connection">
            <i class="fas fa-sync-alt"></i>
          </button>
        `;
      }

      // Debounced search function for better performance
      const debouncedSearch = debounce(runTreeSearch, 300);

      // Clear search function
      function clearSearch() {
        const searchInput = document.getElementById("treeSearchInput");
        const extensionFilter = document.getElementById("extensionFilter");
        const clearBtn = document.getElementById("clearSearchBtn");
        
        searchInput.value = "";
        extensionFilter.value = "";
        clearBtn.style.display = "none";
        
        // Show default tree
        s3Browser.renderCurrentLevel();
      }

      // Show/hide clear button based on search input
      document.getElementById("treeSearchInput").addEventListener("input", function() {
        const clearBtn = document.getElementById("clearSearchBtn");
        clearBtn.style.display = this.value ? "block" : "none";
      });

      // Add keyboard shortcuts for search
      document.getElementById("treeSearchInput").addEventListener("keydown", function(e) {
        if (e.key === "Escape") {
          e.preventDefault();
          clearSearch();
        }
      });

      document
        .getElementById("treeSearchInput")
        .addEventListener("input", debouncedSearch);
      document
        .getElementById("extensionFilter")
        .addEventListener("change", runTreeSearch);

      // Replace the existing loadS3Tree function with:
      async function loadS3Tree() {
        s3Browser.showLoading();
        try {
          // Use lazy loading instead of loading all files at once
          const response = await fetch("/api/s3-lazy-tree?path=&max_items=100");
          const data = await response.json();

          if (response.status === 503) {
            const errorData = await response.json();
            if (errorData.needs_credentials) {
              showS3CredentialsModal();
              return;
            }
          }

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }

          console.log("S3 lazy tree data received:", data);

          // Convert lazy tree data to tree structure
          const treeData = s3Browser.convertLazyTreeToTreeStructure(data);
          s3Browser.rootData = treeData;
          s3Browser.currentData = treeData;
          s3Browser.currentPath = [];
          s3Browser.renderCurrentLevel();
          s3Browser.hideLoading();
        } catch (err) {
          console.error("Failed to load S3 list:", err);
          s3Browser.showError("Failed to load S3 file list. " + err.message);
        }
      }



      // Initialize on page load
      document.addEventListener("DOMContentLoaded", async () => {
        try {
          // Initialize zoom displays
          updateZoomDisplay(1);
          updateZoomDisplay(2);

          // Initialize DICOM viewport layout (default to side-by-side)
          const dicomContainer = document.getElementById(
            "dicomViewportsContainer",
          );
          if (dicomContainer) {
            dicomContainer.classList.add("side-by-side");
          }

          // Initialize S3 viewport layout (default to side-by-side - left and right eye side by side)
          const s3Container = document.getElementById("viewportsContainer");
          if (s3Container) {
            s3Container.classList.add("side-by-side");
            // Force side-by-side layout
            s3Container.style.display = "grid";
            s3Container.style.gridTemplateColumns = "1fr 1fr";
            s3Container.style.gridTemplateRows = "1fr";
          }

          // Initialize S3 status indicator
          updateS3StatusIndicator('checking', 'Initializing...');
          
          // Initialize dedicated scrolling for viewports
          initializeViewportScrolling();

          // Force side-by-side layout immediately
          forceSideBySideLayout();
          
          // Update button text to reflect current state
          const layoutText = document.getElementById("layoutText");
          const layoutIcon = document.getElementById("layoutIcon");
          if (layoutText && layoutIcon) {
            layoutText.textContent = "Switch to Stacked";
            layoutIcon.className = "fas fa-columns";
          }

          console.log("Application initialized with layout controls and sorting");

          // Check if we can hide the overlay immediately (for cases where S3 is already configured)
          setTimeout(() => {
            const overlay = document.getElementById('initialLoadingOverlay');
            if (overlay && overlay.style.display !== 'none') {
              console.log("Quick check: Attempting to hide overlay after basic initialization");
              // Try to hide overlay after basic setup
              hideInitialLoadingOverlay();
            }
          }, 1000);

          // Sequential initialization to prevent race conditions
          await initializeApplication();
        } catch (error) {
          console.error("Error during application initialization:", error);
          updateS3StatusIndicator('error', 'Initialization failed');
          // Ensure loading overlay is hidden even on error
          hideInitialLoadingOverlay();
        }
      });

      // Multiple fallback timeouts to ensure loading overlay is hidden
      setTimeout(() => {
        const overlay = document.getElementById('initialLoadingOverlay');
        if (overlay && overlay.style.display !== 'none') {
          console.log("Fallback 1: Hiding loading overlay after 3 seconds");
          hideInitialLoadingOverlay();
        }
      }, 3000); // 3 second fallback

      setTimeout(() => {
        const overlay = document.getElementById('initialLoadingOverlay');
        if (overlay && overlay.style.display !== 'none') {
          console.log("Fallback 2: Force hiding loading overlay after 5 seconds");
          overlay.style.display = 'none';
          overlay.style.opacity = '0';
        }
      }, 5000); // 5 second force fallback

      setTimeout(() => {
        const overlay = document.getElementById('initialLoadingOverlay');
        if (overlay && overlay.style.display !== 'none') {
          console.log("Fallback 3: Final force hide after 10 seconds");
          overlay.style.display = 'none';
          overlay.style.opacity = '0';
        }
      }, 10000); // 10 second final fallback

      // Sequential initialization function to prevent race conditions
      async function initializeApplication() {
        try {
          // Step 1: Check S3 status
          updateInitialLoadingText("Checking S3 connection...");
          console.log("Step 1: Checking S3 status...");
          const s3Ready = await checkS3Status();
          
          if (!s3Ready) {
            console.log("S3 not ready, showing credentials modal");
            hideInitialLoadingOverlay();
            return;
          }

          // Step 2: Auto-navigate to browse page after S3 is ready
          updateInitialLoadingText("Loading file browser...");
          console.log("Step 2: Navigating to file browser...");
          await toggleFileFrame();
          
          console.log("Application initialization completed successfully");
          // Loading overlay will be hidden by toggleFileFrame after S3 tree loads
        } catch (error) {
          console.error("Error in application initialization:", error);
          updateS3StatusIndicator('error', 'Initialization failed');
          updateInitialLoadingText("Initialization failed. Please refresh the page.");
          
          // Hide loading overlay after a delay to show error
          setTimeout(() => {
            hideInitialLoadingOverlay();
          }, 2000);
          
          // Show a fallback state
          const fileBrowser = document.getElementById("fileBrowserContainer");
          if (fileBrowser) {
            fileBrowser.style.display = "block";
          }
        }
      }

      // Functions to manage initial loading overlay
      function updateInitialLoadingText(text) {
        const loadingText = document.getElementById('initialLoadingText');
        if (loadingText) {
          loadingText.textContent = text;
        }
      }

      function hideInitialLoadingOverlay() {
        const overlay = document.getElementById('initialLoadingOverlay');
        if (overlay) {
          console.log("Hiding initial loading overlay");
          overlay.style.opacity = '0';
          overlay.style.transition = 'opacity 0.3s ease';
          setTimeout(() => {
            overlay.style.display = 'none';
            console.log("Initial loading overlay hidden");
          }, 300);
        } else {
          console.warn("Initial loading overlay not found");
        }
      }

      // Additional function to force hide overlay (for debugging)
      function forceHideInitialLoadingOverlay() {
        const overlay = document.getElementById('initialLoadingOverlay');
        if (overlay) {
          console.log("Force hiding initial loading overlay");
          overlay.style.display = 'none';
          overlay.style.opacity = '0';
          overlay.style.visibility = 'hidden';
        }
      }

      // Make functions available globally for debugging
      window.hideInitialLoadingOverlay = hideInitialLoadingOverlay;
      window.forceHideInitialLoadingOverlay = forceHideInitialLoadingOverlay;

      // Original functions for DICOM upload (keeping for compatibility)
      const crcTable = new Uint32Array(256);
      for (let i = 0; i < 256; i++) {
        let c = i;
        for (let j = 0; j < 8; j++) {
          c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        }
        crcTable[i] = c;
      }

      // Calculate CRC32 from string (for frontend use)
      function calculateCRC32FromString(str) {
        let crc = 0 ^ -1;
        const bytes = new TextEncoder().encode(str);
        for (let i = 0; i < bytes.length; i++) {
          crc = (crc >>> 8) ^ crcTable[(crc ^ bytes[i]) & 0xff];
        }
        return ((crc ^ -1) >>> 0).toString(16).padStart(8, "0");
      }

      async function calculateFileCRC32(file) {
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = function (e) {
            const buffer = e.target.result;
            const array = new Uint8Array(buffer);
            let crc = 0 ^ -1;
            for (let i = 0; i < array.length; i++) {
              crc = (crc >>> 8) ^ crcTable[(crc ^ array[i]) & 0xff];
            }
            const crcValue = ((crc ^ -1) >>> 0).toString(16).padStart(8, "0");
            resolve(crcValue);
          };
          reader.onerror = reject;
          reader.readAsArrayBuffer(file);
        });
      }

      async function uploadDICOM(viewportNumber) {
        const fileInput = document.getElementById(
          "dicomFile" + viewportNumber + "_s3",
        );
        if (fileInput.files.length === 0) return;

        const file = fileInput.files[0];
        console.log(
          "Uploading DICOM file:",
          file.name,
          "to viewport:",
          viewportNumber,
        );
      }

      // Toolbar functionality
      const e2eBtn = document.getElementById("e2eToDicomConverter");
      if (e2eBtn) {
        e2eBtn.addEventListener("click", async function () {
          const fileInput = document.createElement("input");
          fileInput.type = "file";
          fileInput.accept = ".e2e";
          fileInput.onchange = async (event) => {
            const file = event.target.files[0];
            if (!file) return;
            console.log("Converting E2E file:", file.name);
          };
          fileInput.click();
        });
      }

      const metaBtn = document.getElementById("dicomMetadataExtractorTool");
      if (metaBtn) {
        metaBtn.addEventListener("click", async function () {
          const fileInput = document.createElement("input");
          fileInput.type = "file";
          fileInput.accept = ".dcm";
          fileInput.onchange = async (event) => {
            const file = event.target.files[0];
            if (!file) return;
            console.log("Extracting metadata from:", file.name);
          };
          fileInput.click();
        });
      }

      // S3 Credentials Management
      async function checkS3Status() {
        // Update status indicator to checking
        updateS3StatusIndicator('checking', 'Checking connection...');

        if (s3StatusChecked && s3ConfiguredStatus !== null) {
          if (
            !s3ConfiguredStatus.configured &&
            s3ConfiguredStatus.needs_credentials
          ) {
            updateS3StatusIndicator('disconnected', 'Not configured');
            showS3CredentialsModal();
            return false;
          }
          updateS3StatusIndicator('connected', 'Connected');
          return true;
        }

        try {
          const response = await fetch("/api/s3-status", {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
            },
          });

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }

          const status = await response.json();
          s3StatusChecked = true;
          s3ConfiguredStatus = status;

          if (!status.configured && status.needs_credentials) {
            updateS3StatusIndicator('disconnected', 'Credentials required');
            showS3CredentialsModal();
            return false;
          }
          
          updateS3StatusIndicator('connected', `Connected to ${status.bucket}`);
          return true;
        } catch (error) {
          console.error("Error checking S3 status:", error);
          s3StatusChecked = true;
          s3ConfiguredStatus = { configured: false, needs_credentials: true };
          
          // Check if it's a network error vs server error
          if (error.name === 'TypeError' && error.message.includes('fetch')) {
            updateS3StatusIndicator('error', 'Network error');
          } else {
            updateS3StatusIndicator('disconnected', 'Connection error');
          }
          
          // Only show credentials modal for certain types of errors
          if (error.message.includes('401') || error.message.includes('403') || error.message.includes('credentials')) {
            showS3CredentialsModal();
          }
          
          return false;
        }
      }

      function updateS3StatusIndicator(status, text) {
        const statusDot = document.getElementById('s3StatusDot');
        const statusText = document.getElementById('s3StatusText');
        
        if (statusDot && statusText) {
          // Remove all status classes
          statusDot.className = 'status-dot';
          statusText.className = 'status-text';
          
          // Add appropriate classes
          statusDot.classList.add(status);
          statusText.classList.add(status);
          statusText.textContent = text;
        }
      }

      async function refreshS3Status() {
        const refreshBtn = document.querySelector('.status-refresh-btn');
        if (refreshBtn) {
          refreshBtn.classList.add('spinning');
        }
        
        // Reset status cache to force a fresh check
        s3StatusChecked = false;
        s3ConfiguredStatus = null;
        
        try {
          await checkS3Status();
        } finally {
          if (refreshBtn) {
            refreshBtn.classList.remove('spinning');
          }
        }
      }

      function showS3CredentialsModal() {
        document.getElementById("s3CredentialsModal").style.display = "flex";
        document.getElementById("s3ErrorMessage").style.display = "none";
        document.getElementById("s3SuccessMessage").style.display = "none";
      }

      function closeS3Modal() {
        document.getElementById("s3CredentialsModal").style.display = "none";
      }

      // Empty file warning modal functions
      function showEmptyFileWarning() {
        document.getElementById("emptyFileWarningModal").style.display = "flex";
      }

      function closeEmptyFileWarning() {
        document.getElementById("emptyFileWarningModal").style.display = "none";
        // Reset E2E mode since the file is empty
        resetE2EMode();
      }

      async function submitS3Credentials() {
        const accessKey = document.getElementById("s3AccessKey").value.trim();
        const secretKey = document.getElementById("s3SecretKey").value.trim();
        const region = document.getElementById("s3Region").value;
        const bucket = document.getElementById("s3Bucket").value.trim();
        const saveToEnv = document.getElementById("s3SaveToEnv").checked;

        // Enhanced validation
        if (!accessKey || !secretKey || !region || !bucket) {
          showS3Error("Please fill in all fields.");
          return;
        }

        // Validate access key format
        if (!accessKey.startsWith('AKIA') && accessKey.length !== 20) {
          showS3Error("Invalid Access Key ID format. Should start with 'AKIA' and be 20 characters long.");
          return;
        }

        // Validate secret key length
        if (secretKey.length < 40) {
          showS3Error("Invalid Secret Access Key. Should be at least 40 characters long.");
          return;
        }

        // Validate bucket name
        if (bucket.length < 3 || bucket.length > 63) {
          showS3Error("Invalid bucket name. Must be between 3 and 63 characters long.");
          return;
        }

        // Show loading state
        document.getElementById("s3FormContent").style.display = "none";
        document.getElementById("s3LoadingState").style.display = "block";
        document.getElementById("s3ErrorMessage").style.display = "none";
        document.getElementById("s3SuccessMessage").style.display = "none";

        try {
          console.log("Submitting S3 credentials...");
          const response = await fetch("/api/set-s3-credentials", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              accessKey: accessKey,
              secretKey: secretKey,
              region: region,
              bucket: bucket,
              saveToEnv: saveToEnv,
            }),
          });

          const result = await response.json();
          console.log("S3 credentials response:", result);

          if (response.ok) {
            const successMessage = result.message + 
              (result.saved_to_env ? " Credentials saved to .env file." : " Credentials set for this session only.");
            
            showS3Success(successMessage);

            // Reset status cache
            s3StatusChecked = false;
            s3ConfiguredStatus = null;
            s3TreeData = null;

            // Update status indicator
            updateS3StatusIndicator('connected', `Connected to ${bucket}`);

            // Close modal and reload tree after delay
            setTimeout(() => {
              closeS3Modal();
              loadS3Tree();
            }, 2000);
          } else {
            // Handle different types of errors
            let errorMessage = "Failed to set credentials";
            if (result.detail) {
              errorMessage = result.detail;
            } else if (result.error) {
              errorMessage = result.error;
            }
            
            // Provide more helpful error messages
            if (errorMessage.includes("Invalid credentials")) {
              errorMessage = "Invalid AWS credentials. Please check your Access Key ID and Secret Access Key.";
            } else if (errorMessage.includes("bucket")) {
              errorMessage = "Bucket not found or not accessible. Please check the bucket name and your permissions.";
            } else if (errorMessage.includes("region")) {
              errorMessage = "Invalid region. Please select the correct AWS region for your bucket.";
            }
            
            throw new Error(errorMessage);
          }
        } catch (error) {
          console.error("Error setting S3 credentials:", error);
          
          // Handle network errors
          if (error.name === 'TypeError' && error.message.includes('fetch')) {
            showS3Error("Network error. Please check your connection and try again.");
          } else {
            showS3Error("Error: " + error.message);
          }
        } finally {
          document.getElementById("s3LoadingState").style.display = "none";
          document.getElementById("s3FormContent").style.display = "block";
        }
      }

      function showS3Error(message) {
        const errorDiv = document.getElementById("s3ErrorMessage");
        errorDiv.textContent = message;
        errorDiv.style.display = "block";
        document.getElementById("s3SuccessMessage").style.display = "none";
      }

      function showS3Success(message) {
        const successDiv = document.getElementById("s3SuccessMessage");
        successDiv.textContent = message;
        successDiv.style.display = "block";
        document.getElementById("s3ErrorMessage").style.display = "none";
      }

      // Add responsive resize handler
      window.addEventListener("resize", () => {
        // Recalculate constraints for both viewports
        for (let viewportNumber of [1, 2]) {
          if (viewportData[viewportNumber]) {
            updateImageTransform(viewportNumber);
          }
        }
      });

      // Show context menu
      function showContextMenu(event, filePath) {
        // Prevent the default browser context menu
        event.preventDefault();

        // Set the selected file path for menu actions
        selectedFilePath = filePath;

        const contextMenu = document.getElementById("contextMenu");
        const loadE2EOption = document.getElementById("loadE2EOption");
        const viewport1Option = document.getElementById("viewport1Option");
        const viewport2Option = document.getElementById("viewport2Option");
        const downloadOption = document.getElementById("downloadOption");
        if (!contextMenu) return;

        // Show/hide options based on file type
        const fileType = detectFileType(filePath);
        const canDownload = ["DICOM", "E2E", "FDA", "FDS"].includes(fileType);
        const canView = fileType === "DICOM";
        
        if (loadE2EOption) {
          loadE2EOption.style.display = fileType === "E2E" ? "block" : "none";
        }
        
        if (viewport1Option) {
          viewport1Option.style.display = canView ? "block" : "none";
        }
        
        if (viewport2Option) {
          viewport2Option.style.display = canView ? "block" : "none";
        }
        
        if (downloadOption) {
          downloadOption.style.display = canDownload ? "block" : "none";
        }

        // Make the menu visible to measure its dimensions
        contextMenu.style.display = "block";
        const menuWidth = contextMenu.offsetWidth;
        const menuHeight = contextMenu.offsetHeight;

        const pageWidth = window.innerWidth;
        const pageHeight = window.innerHeight;

        // Get cursor position from the event object
        let x = event.pageX;
        let y = event.pageY;

        // Adjust position if the menu would go off-screen
        if (x + menuWidth > pageWidth) {
          x = pageWidth - menuWidth - 10; // Add a small margin
        }
        if (y + menuHeight > pageHeight) {
          y = pageHeight - menuHeight - 10; // Add a small margin
        }

        // Apply the calculated position
        contextMenu.style.left = `${x}px`;
        contextMenu.style.top = `${y}px`;
      }

      document
        .getElementById("fileTreeContainer")
        .addEventListener("contextmenu", (e) => {
          console.log("Context menu event triggered");
          
          // Find the nearest .tree-item element
          const treeItem = e.target.closest(".tree-item");
          console.log("Tree item found:", treeItem);

          // Ensure it's a file item with a path
          if (!treeItem || !treeItem.dataset.filePath) {
            console.log("No tree item or file path found");
            return;
          }

          // Get the file path from the data attribute
          const filePath = treeItem.dataset.filePath;
          console.log("File path from context menu:", filePath);

          // Call our standardized function to show the menu
          showContextMenu(e, filePath);

          // Optional: Highlight the selected item
          document
            .querySelectorAll(".tree-item.selected")
            .forEach((el) => el.classList.remove("selected"));
          if (treeItem.parentElement.matches("li.tree-item")) {
            // check if it's the li or the div inside it
            treeItem.parentElement.classList.add("selected");
          } else {
            treeItem.classList.add("selected");
          }
        });

      // Hide context menu on click outside
      document.addEventListener("click", (e) => {
        const menu = document.getElementById("contextMenu");
        if (!menu) return;
        if (!menu.contains(e.target)) {
          menu.style.display = "none";
        }
      });

      // Load DICOM image into viewport from context menu
        async function loadIntoViewport(viewportNumber) {
            if (!selectedFilePath) {
                        console.error("No file path selected");
        showNotification("Please select a DICOM file first.", "warning");
        return;
            }

            // Clear the viewport before loading new file
            clearViewport(viewportNumber);

            // Show progress bar right away
            showProgress(
                viewportNumber,
                "Preparing to load...",
                { File: selectedFilePath.split("/").pop() }
            );

            document.getElementById("contextMenu").style.display = "none";

            try {
                await loadIntoViewportWithPath(viewportNumber, selectedFilePath);
            } catch (error) {
                console.error("Error in loadIntoViewport:", error);
                showNotification(`Error loading file: ${error.message}`, "error");
                hideProgress(viewportNumber);
            }
     }
      //ensurefilechached 

      async function downloadSelectedFile() {
        try {
          if (!selectedFilePath) {
            showNotification("Please select a file to download.", "warning");
            return;
          }
          const fileType = detectFileType(selectedFilePath);
          const supported = ["DICOM", "E2E", "FDA", "FDS"].includes(fileType);
          if (!supported) {
            showNotification(`Unsupported file type for download: ${fileType}`, "warning");
            return;
          }
          const url = `/api/s3-download?path=${encodeURIComponent(selectedFilePath)}`;
          const a = document.createElement("a");
          a.href = url;
          a.download = selectedFilePath.split("/").pop();
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          document.getElementById("contextMenu").style.display = "none";
          showNotification("Download started. Check your browser downloads.", "info");
        } catch (err) {
          console.error("Download error:", err);
          showNotification(`Download failed: ${err.message}`, "error");
        }
      }

      async function ensureFileCached(filePath) {
        let crc = fileCRCs[filePath];
        if (!crc) {
          // Check if the file is already cached in the backend
          try {
            const checkResponse = await fetch(`/api/download_dicom_from_s3?path=${encodeURIComponent(filePath)}`);
            if (checkResponse.ok) {
              const checkData = await checkResponse.json();
              if (checkData.cache_source === "memory" || checkData.cache_source === "disk") {
                // File is already cached, get the CRC from the response
                console.log(`File ${filePath} is already cached (${checkData.cache_source})`);
                // Generate a CRC for this file path to store locally
                const pathCRC = calculateCRC32FromString(filePath);
                fileCRCs[filePath] = pathCRC;
                return pathCRC;
              }
            }
          } catch (error) {
            console.warn(`Could not check if file is cached: ${error.message}`);
          }

          // File not cached, download it normally (no pre-caching)
          try {
            const response = await fetch(`/api/download_dicom_from_s3?path=${encodeURIComponent(filePath)}`);
            if (response.ok) {
              const data = await response.json();
              // Generate a CRC for this file path to store locally
              const pathCRC = calculateCRC32FromString(filePath);
              fileCRCs[filePath] = pathCRC;
              return pathCRC;
            } else {
              throw new Error("Failed to download file for viewing.");
            }
          } catch (error) {
            console.error(`Error downloading file ${filePath}:`, error);
            throw new Error("Failed to download file for viewing.");
          }
        }
        return crc;
      }

      

      // Enhanced frame loading with CRC caching and cancellation support
      async function loadFrameWithCRC(
        viewportNumber,
        frameNumber,
        abortController = null,
      ) {
        const img = document.getElementById(`viewportImage${viewportNumber}`);
        const data = viewportData[viewportNumber];

        if (!data) {
          console.error("No DICOM data available for viewport", viewportNumber);
          return;
        }

        try {
          console.log(
            `Loading frame ${frameNumber} for viewport ${viewportNumber} with CRC caching`,
          );

          // Check if operation was cancelled
          if (abortController && abortController.signal.aborted) {
            throw new Error("Operation cancelled by user");
          }

          // Create metadata for this specific frame
          const frameMetadata = {
            path: data.s3_key,
            dicomFilePath: data.dicom_file_path,
            frame: frameNumber,
            size: data.size || 0,
            lastModified: data.last_modified || "",
          };
          
          // Debug: Log the dicom_file_path being used
          console.log(`[DEBUG] loadFrameWithCRC - viewport ${viewportNumber}, frame ${frameNumber}`);
          console.log(`[DEBUG] data.dicom_file_path: ${data.dicom_file_path}`);
          console.log(`[DEBUG] data.s3_key: ${data.s3_key}`);
          console.log(`[DEBUG] frameMetadata.dicomFilePath: ${frameMetadata.dicomFilePath}`);

          // Try CRC cache first
          try {
            const cachedResult = await loadImageWithCRC(
              data.s3_key,
              frameMetadata,
            );

            // Check if cancelled after cache check
            if (abortController && abortController.signal.aborted) {
              throw new Error("Operation cancelled by user");
            }

            img.onload = () => {
              console.log(
                `Frame ${frameNumber} loaded successfully with CRC caching (${cachedResult.source})`,
              );
              img.style.display = "block";
              img.style.maxWidth = "100%";
              img.style.maxHeight = "100%";
              img.style.width = "auto";
              img.style.height = "auto";

              resetZoom(viewportNumber);
              setupImageInteractions(viewportNumber);
              updateFrameInfo(viewportNumber, frameNumber);
            };

            img.onerror = (error) => {
              console.error(
                `Failed to load cached image for frame ${frameNumber}`,
                error
              );
              throw new Error(`Failed to load cached image: ${error.message || 'Unknown error'}`);
            };

            img.src = cachedResult.imageData;
            currentFrames[viewportNumber] = frameNumber;

            console.log(
              `Frame ${frameNumber} loaded from ${cachedResult.source} with CRC: ${cachedResult.cacheKey.substring(0, 8)}`,
            );
            return;
          } catch (cacheError) {
            if (abortController && abortController.signal.aborted) {
              throw new Error("Operation cancelled by user");
            }
            console.warn(
              `CRC cache failed for frame ${frameNumber}: ${cacheError.message}`,
            );
            // Fall back to original method
          }

          // Fallback to original PNG endpoint
          console.log(`[DEBUG] Fallback method - using data.dicom_file_path: ${data.dicom_file_path}`);
          const fetchOptions = abortController
            ? { signal: abortController.signal }
            : {};
          const pngResponse = await fetch(
            `/api/view_dicom_png?frame=${frameNumber}&dicom_file_path=${encodeURIComponent(data.dicom_file_path)}&v=${Date.now()}`,
            fetchOptions,
          );

          if (!pngResponse.ok) {
            throw new Error(`Failed to get PNG: ${pngResponse.statusText}`);
          }

          const pngBlob = await pngResponse.blob();
          const pngUrl = URL.createObjectURL(pngBlob);

          // Check if cancelled after download
          if (abortController && abortController.signal.aborted) {
            throw new Error("Operation cancelled by user");
          }

          // Cache this frame for future use
          try {
            imageCache.set(
              imageCache.generateCacheKey(data.s3_key, frameMetadata),
              pngUrl,
              frameMetadata,
            );
          } catch (cacheSetError) {
            console.warn(
              `Failed to cache frame ${frameNumber}: ${cacheSetError.message}`,
            );
          }

          // Display the image with proper constraints
          img.onload = () => {
            console.log(
              `Image loaded successfully for viewport ${viewportNumber} (fallback method)`,
            );
            img.style.display = "block";
            img.style.maxWidth = "100%";
            img.style.maxHeight = "100%";
            img.style.width = "auto";
            img.style.height = "auto";

            centerImage(viewportNumber);
            setupImageInteractions(viewportNumber);
            updateFrameInfo(viewportNumber, frameNumber);
          };

          img.onerror = (error) => {
            console.error(`Failed to load image for frame ${frameNumber}`, error);
            throw new Error(`Failed to load image: ${error.message || 'Unknown error'}`);
          };

          img.src = pngUrl;
          currentFrames[viewportNumber] = frameNumber;
        } catch (error) {
          console.error("Error loading frame:", error);
          throw error;
        }
      }

      // Load specific frame (updated to use CRC caching)
      async function loadFrame(viewportNumber, frameNumber) {
        return await loadFrameWithCRC(viewportNumber, frameNumber);
      }

      // Debug function to check DICOM file status
      async function checkDicomFileStatus(dicomFilePath) {
        try {
          console.log(`[DEBUG] Checking DICOM file status: ${dicomFilePath}`);
          
          // Check if file is ready
          const readyResponse = await fetch(`/api/check_dicom_ready?dicom_file_path=${encodeURIComponent(dicomFilePath)}`);
          const readyData = await readyResponse.json();
          console.log(`[DEBUG] DICOM ready status:`, readyData);
          
          // Check file info
          const infoResponse = await fetch(`/api/file_info/${encodeURIComponent(dicomFilePath)}`);
          if (infoResponse.ok) {
            const infoData = await infoResponse.json();
            console.log(`[DEBUG] DICOM file info:`, infoData);
          } else {
            console.log(`[DEBUG] Could not get file info: ${infoResponse.status}`);
          }
          
          return readyData;
        } catch (error) {
          console.error(`[DEBUG] Error checking DICOM status:`, error);
          return { ready: false, error: error.message };
        }
      }

      // Function to clear cache and retry loading
      async function clearCacheAndRetry(viewportNumber, filePath) {
        try {
          const fileType = detectFileType(filePath);
          console.log(`[CACHE CLEAR] Clearing cache and retrying for ${filePath} (${fileType})`);
          
          // Clear the image cache
          imageCache.clear();
          
          // Clear file CRCs
          delete fileCRCs[filePath];
          
          // Clear viewport data
          viewportData[viewportNumber] = {};
          
          // For E2E files, also clear the tree data
          if (fileType === 'E2E') {
            console.log(`[CACHE CLEAR] Clearing E2E tree data`);
            // Clear any E2E-specific data
            const leftEyeTree = document.getElementById('leftEyeTree');
            const rightEyeTree = document.getElementById('rightEyeTree');
            if (leftEyeTree) leftEyeTree.innerHTML = '';
            if (rightEyeTree) rightEyeTree.innerHTML = '';
          }
          
          // Retry loading
          await loadIntoViewportWithPath(viewportNumber, filePath);
        } catch (error) {
          console.error(`[CACHE CLEAR] Error during retry:`, error);
          throw error;
        }
      }

      // Function to clear all caches (manual debug function)
      async function clearAllCaches() {
        try {
          console.log(`[CACHE CLEAR] Clearing all caches`);
          
          // Clear the image cache
          imageCache.clear();
          
          // Clear all file CRCs
          Object.keys(fileCRCs).forEach(key => delete fileCRCs[key]);
          
          // Clear all viewport data
          viewportData[1] = {};
          viewportData[2] = {};
          
          // Clear viewport displays
          clearViewport(1);
          clearViewport(2);
          
          // Clear backend cache
          try {
            const response = await fetch('/api/clear-cache', { method: 'POST' });
            if (response.ok) {
              const result = await response.json();
              console.log(`[CACHE CLEAR] Backend cache cleared:`, result);
            } else {
              console.warn(`[CACHE CLEAR] Failed to clear backend cache: ${response.status}`);
            }
          } catch (backendError) {
            console.warn(`[CACHE CLEAR] Backend cache clear failed:`, backendError);
          }
          
          // Show notification
          showNotification("All caches cleared successfully", "success");
          
          console.log(`[CACHE CLEAR] All caches cleared`);
        } catch (error) {
          console.error(`[CACHE CLEAR] Error clearing caches:`, error);
          showNotification("Error clearing caches", "error");
        }
      }

      // Function to initialize dedicated scrolling for entire viewports
      function initializeViewportScrolling() {
        const viewport1 = document.getElementById('viewport1');
        const viewport2 = document.getElementById('viewport2');
        
        if (viewport1) {
          // Store scroll position for viewport 1 (left eye)
          viewport1.addEventListener('scroll', function() {
            sessionStorage.setItem('viewport1ScrollX', viewport1.scrollLeft);
            sessionStorage.setItem('viewport1ScrollY', viewport1.scrollTop);
          });
          
          // Restore scroll position for viewport 1
          const savedScrollX1 = sessionStorage.getItem('viewport1ScrollX');
          const savedScrollY1 = sessionStorage.getItem('viewport1ScrollY');
          if (savedScrollX1 && savedScrollY1) {
            viewport1.scrollLeft = parseInt(savedScrollX1);
            viewport1.scrollTop = parseInt(savedScrollY1);
          }
        }
        
        if (viewport2) {
          // Store scroll position for viewport 2 (right eye)
          viewport2.addEventListener('scroll', function() {
            sessionStorage.setItem('viewport2ScrollX', viewport2.scrollLeft);
            sessionStorage.setItem('viewport2ScrollY', viewport2.scrollTop);
          });
          
          // Restore scroll position for viewport 2
          const savedScrollX2 = sessionStorage.getItem('viewport2ScrollX');
          const savedScrollY2 = sessionStorage.getItem('viewport2ScrollY');
          if (savedScrollX2 && savedScrollY2) {
            viewport2.scrollLeft = parseInt(savedScrollX2);
            viewport2.scrollTop = parseInt(savedScrollY2);
          }
        }
      }

      // Function to reset scroll positions for a specific viewport
      function resetViewportScroll(viewportNumber) {
        const viewport = document.getElementById(`viewport${viewportNumber}`);
        if (viewport) {
          viewport.scrollLeft = 0;
          viewport.scrollTop = 0;
          sessionStorage.removeItem(`viewport${viewportNumber}ScrollX`);
          sessionStorage.removeItem(`viewport${viewportNumber}ScrollY`);
        }
      }

      // Function to force stacked layout
      function forceStackedLayout() {
        const container = document.getElementById("viewportsContainer");
        if (container) {
          // Remove any side-by-side classes
          container.classList.remove("side-by-side");
          
          // Force stacked layout with inline styles
          container.style.display = "grid";
          container.style.gridTemplateColumns = "1fr";
          container.style.gridTemplateRows = "1fr 1fr";
          container.style.gap = "10px";
          container.style.height = "calc(100vh - 180px)";
          container.style.minHeight = "calc(100vh - 180px)";
          
          // Update the state
          isStackedLayout = true;
          
          console.log("Forced stacked layout applied");
        }
      }

      // Function to force side-by-side layout
      function forceSideBySideLayout() {
        const container = document.getElementById("viewportsContainer");
        if (container) {
          // Add side-by-side class
          container.classList.add("side-by-side");
          
          // Force side-by-side layout with inline styles
          container.style.display = "grid";
          container.style.gridTemplateColumns = "1fr 1fr";
          container.style.gridTemplateRows = "1fr";
          container.style.gap = "10px";
          container.style.height = "auto";
          container.style.minHeight = "calc(100vh - 180px)";
          
          // Update the state
          isStackedLayout = false;
          
          console.log("Forced side-by-side layout applied");
        }
      }

      // Function to show progress for .EX.DCM files when clicked
      async function showEXDCMProgress(fileItem) {
        try {
          console.log(`[EX.DCM] Showing progress for file: ${fileItem.name}`);
          
          // Show progress in viewport 1 by default (or the first available viewport)
          const viewportNumber = 1;
          
          // Extract file info for metadata
          const fileName = fileItem.name;
          const filePath = fileItem.path;
          
          // Show enhanced progress immediately
          showProgress(
            viewportNumber,
            "Preparing .EX.DCM file...",
            {
              File: fileName,
              Type: "EX.DCM",
              Extension: "EX.DCM",
              Status: "Initializing..."
            }
          );
          
          // Small delay to ensure progress is visible
          await new Promise(resolve => setTimeout(resolve, 200));
          
          // Automatically load the file into the viewport
          // The loadIntoViewportWithPath function will now handle the "WORK IN PROGRESS" response
          await loadIntoViewportWithPath(viewportNumber, filePath);
          
        } catch (error) {
          console.error(`[EX.DCM] Error showing progress:`, error);
          showNotification(`Error processing .EX.DCM file: ${error.message}`, "error");
          
          // Hide progress on error
          hideProgress(1);
        }
      }

      // Function to pre-cache files (manual pre-cache function)
      async function preCacheFiles() {
        try {
          console.log(`[PRE-CACHE] Starting pre-cache operation`);
          
          // Check if there are selected files first
          let filesToCache = [];
          
          if (selectedFiles && selectedFiles.size > 0) {
            // Use selected files
            filesToCache = Array.from(selectedFiles);
            console.log(`[PRE-CACHE] Using ${filesToCache.length} selected files:`, filesToCache);
          } else {
            // Fallback: search all files in the tree
            console.log(`[PRE-CACHE] No files selected, searching all files in tree`);
            
            // Get the current file tree data
            const fileTreeContainer = document.getElementById('fileTreeContainer');
            if (!fileTreeContainer) {
              throw new Error('File tree container not found');
            }
            
            // Find all files in the tree (not just DICOM files)
            const allFiles = [];
            const findAllFiles = (element) => {
              // Check if this element is a file item
              if (element.classList && element.classList.contains('file-item')) {
                const fileName = element.textContent || element.innerText;
                // Get file path for any file type
                const filePath = element.getAttribute('data-filePath') || 
                                element.getAttribute('data-path') || 
                                element.getAttribute('data-fullPath');
                if (filePath) {
                  allFiles.push(filePath);
                  console.log(`[PRE-CACHE] Found file: ${fileName} -> ${filePath}`);
                } else {
                  console.warn(`[PRE-CACHE] No path found for file: ${fileName}`);
                }
              }
              // Recursively search children
              if (element.children) {
                for (let child of element.children) {
                  findAllFiles(child);
                }
              }
            };
            
            findAllFiles(fileTreeContainer);
            
            console.log(`[PRE-CACHE] Found ${allFiles.length} files to pre-cache:`, allFiles);
            
            // Fallback: if no files found in DOM, try to get from s3Browser data
            if (allFiles.length === 0 && window.s3Browser && window.s3Browser.rootData) {
              console.log(`[PRE-CACHE] Trying fallback method with s3Browser data`);
              const fallbackFiles = [];
              const extractFilesFromTree = (nodes) => {
                nodes.forEach(node => {
                  if (node.type === 'file' && node.path) {
                    fallbackFiles.push(node.path);
                    console.log(`[PRE-CACHE] Fallback found file: ${node.name} -> ${node.path}`);
                  }
                  if (node.children) {
                    extractFilesFromTree(node.children);
                  }
                });
              };
              extractFilesFromTree(window.s3Browser.rootData);
              allFiles.push(...fallbackFiles);
            }
            
            filesToCache = allFiles;
          }
          
          if (filesToCache.length === 0) {
            showNotification("No files found to pre-cache. Please select files using the checkboxes first, or ensure files are loaded in the tree.", "warning");
            return;
          }
          
          // Show the progress popup
          showCacheProgressPopup(filesToCache.length);
          
          // Initialize counters
          let successful = 0;
          let failed = 0;
          let skipped = 0;
          let currentIndex = 0;
          
          // Process files one by one to show real-time progress
          for (const filePath of filesToCache) {
            currentIndex++;
            const fileName = filePath.split('/').pop();
            
            // Update current file display
            updateCurrentFile(fileName, currentIndex, filesToCache.length);
            updateOverallProgress((currentIndex - 1) / filesToCache.length * 100, currentIndex - 1, filesToCache.length);
            
            try {
              console.log(`[PRE-CACHE] Processing file ${currentIndex}/${filesToCache.length}: ${filePath}`);
              
              // Update status to downloading
              updateDownloadStatus(fileName, 'Starting download...');
              
              // Update to processing status after a short delay
              setTimeout(() => {
                updateProcessingStatus(fileName, 'Processing file...');
              }, 1000);
              
              // Use the pre-cache endpoint for individual file
              const response = await fetch('/api/pre-cache-files', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  file_paths: [filePath]
                })
              });
              
              if (response.ok) {
                const result = await response.json();
                console.log(`[PRE-CACHE] File processed:`, result);
                console.log(`[PRE-CACHE] Response structure:`, {
                  hasMessage: !!result.message,
                  message: result.message,
                  hasResults: !!result.results,
                  hasProcessedFiles: !!(result.results && result.results.processed_files),
                  processedFilesLength: result.results?.processed_files?.length || 0
                });
                
                // Since the response was ok, assume success (the backend processing functions return JSONResponse on success)
                successful++;
                updateFileComplete(fileName, 'success', result.dicom_file_path || result.crc || 'N/A');
                console.log(`[PRE-CACHE] Successfully cached: ${fileName}`);
                
              } else {
                const errorText = await response.text();
                console.error(`[PRE-CACHE] Server error for ${fileName}: ${response.status} - ${errorText}`);
                failed++;
                updateFileError(fileName, `Server error: ${response.status}`);
              }
              
            } catch (fileError) {
              console.error(`[PRE-CACHE] Error processing ${fileName}:`, fileError);
              failed++;
              updateFileError(fileName, fileError.message || 'Network error');
            }
            
            // Update overall progress
            updateOverallProgress(currentIndex / filesToCache.length * 100, currentIndex, filesToCache.length);
            
            // Add small delay between files
            await new Promise(resolve => setTimeout(resolve, 200));
          }
          
          // Show completion
          handleCachingComplete(filesToCache.length, successful);
          
          // Show final notification
          let message = `Pre-cache completed: ${successful}/${filesToCache.length} files processed`;
          if (skipped > 0) {
            message += `, ${skipped} already cached`;
          }
          if (failed > 0) {
            message += `, ${failed} failed`;
          }
          
          showNotification(message, failed > 0 ? "warning" : "success");
          
        } catch (error) {
          console.error(`[PRE-CACHE] Error during pre-cache operation:`, error);
          showNotification("Error during pre-cache operation", "error");
          hideCacheProgressPopup();
        }
      }

      // Setup frame slider - handle both regular frames and OCT frames
      function setupFrameSlider(viewportNumber) {
        console.log(`Setting up frame slider for viewport ${viewportNumber}`);

        const frameSlider = document.getElementById(
          `frameSlider${viewportNumber}`,
        );
        const frameSliderContainer = document.getElementById(
          `frameSliderContainer${viewportNumber}`,
        );

        if (!frameSlider || !frameSliderContainer) {
          console.error(
            "Frame slider elements not found for viewport",
            viewportNumber,
          );
          return;
        }

        // Get the appropriate eye frame data
        const eyeData = viewportNumber === 1 ? leftEyeFrameData : rightEyeFrameData;
        const data = viewportData[viewportNumber];

        if (!eyeData || !data) {
          console.warn(`No eye data available for viewport ${viewportNumber}`);
          return;
        }

        // Hide frame slider for flattened OCT images
        if (data && data.isFlattened) {
          frameSliderContainer.classList.remove("active");
          console.log(
            `Frame slider hidden for flattened OCT image in viewport ${viewportNumber}`,
          );
          return;
        }

        // Determine frame count based on mode and eye data
        let numFrames;
        let isOctMode = eyeData.isOctMode;
        
        if (isOctMode && eyeData.octFrameCount > 0) {
          // OCT mode - use original OCT frame count
          numFrames = eyeData.octFrameCount;
          console.log(`OCT mode detected for viewport ${viewportNumber}, ${numFrames} OCT frames`);
        } else {
          // Regular mode - use total frames
          numFrames = eyeData.totalFrames;
          console.log(`Regular mode for viewport ${viewportNumber}, ${numFrames} frames`);
        }

        // Always reset to first frame when setting up slider for new sub-branch
        frameSlider.min = 0;
        frameSlider.max = Math.max(0, numFrames - 1);
        frameSlider.value = 0; // Always start at first frame (0)
        
        // Reset current frame in eye data
        eyeData.currentFrame = 0;
        
        console.log(`Slider reset to first frame (0) for viewport ${viewportNumber}, range: 0 to ${numFrames - 1}`);

        // Remove existing event listeners to prevent conflicts
        frameSlider.onchange = null;
        frameSlider.oninput = null;

        if (numFrames <= 1) {
          frameSlider.disabled = true;
          console.log(
            `Single-frame detected for viewport ${viewportNumber}, slider disabled`,
          );
        } else {
          frameSlider.disabled = false;
          
          // Use a debounced event handler for better performance
          let sliderTimeout;
          frameSlider.oninput = async (e) => {
            const frameNumber = parseInt(e.target.value);
            console.log(
              `Frame slider changed to ${frameNumber} for viewport ${viewportNumber} (${isOctMode ? 'OCT' : 'regular'} mode)`,
            );
            
            // Update eye data current frame
            eyeData.currentFrame = frameNumber;
            
            // Update UI immediately for responsiveness
            updateFrameInfo(viewportNumber, frameNumber);
            updateNavigationButtons(viewportNumber, frameNumber, numFrames);
            
            // Clear previous timeout
            if (sliderTimeout) {
              clearTimeout(sliderTimeout);
            }
            
            // Debounce the actual frame loading
            sliderTimeout = setTimeout(async () => {
              try {
                if (isOctMode && data && data.dicom_file_path) {
                  // OCT mode - load original OCT frame
                  const eye = viewportNumber === 1 ? 'left' : 'right';
                  await loadOriginalOCTFrame(viewportNumber, frameNumber, data.dicom_file_path, eye);
                } else if (isE2EMode && data && data.dicom_file_path) {
                  // E2E mode - load E2E eye image
                  const eye = viewportNumber === 1 ? 'left' : 'right';
                  await loadE2EEyeImage(viewportNumber, frameNumber, data.dicom_file_path, eye);
                } else {
                  // Regular mode - load regular frame
                  await loadFrame(viewportNumber, frameNumber);
                }
              } catch (error) {
                console.error("Error loading frame from slider:", error);
                // Revert slider on error
                frameSlider.value = eyeData.currentFrame;
                updateFrameInfo(viewportNumber, eyeData.currentFrame);
                updateNavigationButtons(viewportNumber, eyeData.currentFrame, numFrames);
              }
            }, 150); // 150ms debounce
          };
        }

        frameSliderContainer.classList.add("active");
        updateFrameInfo(viewportNumber, 0); // Always show "Frame 1 of N" when setting up

        // Update navigation button states
        updateNavigationButtons(viewportNumber, 0, numFrames);

        // Show mode switch controls for E2E files with OCT frames
        const modeSwitchControls = document.getElementById(`modeSwitchControls${viewportNumber}`);
        if (modeSwitchControls && eyeData.octFrameCount > 0 && isE2EMode) {
          modeSwitchControls.style.display = "flex";
          
          // Update button states based on current mode
          const regularBtn = modeSwitchControls.querySelector('button[onclick*="switchToRegularMode"]');
          const octBtn = modeSwitchControls.querySelector('button[onclick*="switchToOCTMode"]');
          
          if (regularBtn && octBtn) {
            if (isOctMode) {
              regularBtn.classList.remove('active');
              octBtn.classList.add('active');
            } else {
              regularBtn.classList.add('active');
              octBtn.classList.remove('active');
            }
          }
          
          console.log(`Mode switch controls shown for viewport ${viewportNumber}, OCT frames: ${eyeData.octFrameCount}, current mode: ${isOctMode ? 'OCT' : 'Regular'}`);
        } else if (modeSwitchControls) {
          modeSwitchControls.style.display = "none";
          console.log(`Mode switch controls hidden for viewport ${viewportNumber} - OCT frames: ${eyeData.octFrameCount || 0}, E2E mode: ${isE2EMode}`);
        }

        // Show frame input controls for OCT frames (when in OCT mode or for multi-frame DICOM)
        const frameInputControls = document.getElementById(`frameInputControls${viewportNumber}`);
        if (frameInputControls) {
          const shouldShowFrameInput = (isOctMode && eyeData.octFrameCount > 0) || 
                                     (numFrames > 1 && !isE2EMode); // Show for multi-frame DICOM too
          
          if (shouldShowFrameInput) {
            frameInputControls.style.display = "flex";
            
            // Update the input field max value and current value
            const frameInput = document.getElementById(`frameInput${viewportNumber}`);
            if (frameInput) {
              const maxFrame = isOctMode ? eyeData.octFrameCount : numFrames;
              frameInput.max = maxFrame;
              frameInput.value = parseInt(frameSlider.value) + 1; // Convert from 0-based to 1-based
              
              // Ensure event listener is attached when frame input is shown
              setupFrameInputListeners();
            }
            
            console.log(`Frame input controls shown for viewport ${viewportNumber}, max frame: ${isOctMode ? eyeData.octFrameCount : numFrames}`);
          } else {
            frameInputControls.style.display = "none";
            console.log(`Frame input controls hidden for viewport ${viewportNumber}`);
          }
        }

        console.log(
          `Frame slider setup complete for viewport ${viewportNumber}, ${numFrames} frame(s) (${isOctMode ? 'OCT' : 'regular'} mode)`,
        );
        console.log(`Viewport ${viewportNumber} eye data:`, eyeData);
      }

      // Update navigation button states
      function updateNavigationButtons(viewportNumber, currentFrame, totalFrames) {
        const prevBtn = document.getElementById(`framePrev${viewportNumber}`);
        const nextBtn = document.getElementById(`frameNext${viewportNumber}`);
        
        if (prevBtn && nextBtn) {
          // Enable/disable buttons based on frame count
          const hasMultipleFrames = totalFrames > 1;
          prevBtn.disabled = !hasMultipleFrames;
          nextBtn.disabled = !hasMultipleFrames;
          
          // Add visual feedback for single frame
          if (!hasMultipleFrames) {
            prevBtn.title = "Single frame - no navigation available";
            nextBtn.title = "Single frame - no navigation available";
          } else {
            prevBtn.title = "Previous Frame (â†)";
            nextBtn.title = "Next Frame (â†’)";
          }
        }
      }

      // Add keyboard navigation support
      document.addEventListener('keydown', function(event) {
        // Handle Enter key in frame input fields
        if (event.target.tagName === 'INPUT' && event.target.id && event.target.id.startsWith('frameInput') && event.key === 'Enter') {
          event.preventDefault();
          const viewportNumber = event.target.id.replace('frameInput', '');
          goToFrame(parseInt(viewportNumber));
          return;
        }

        // Only handle navigation if not typing in an input field
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
          return;
        }

        const activeViewport = getActiveViewport();
        if (!activeViewport) return;

        switch(event.key) {
          case 'ArrowLeft':
            event.preventDefault();
            navigateFrame(activeViewport, -1);
            break;
          case 'ArrowRight':
            event.preventDefault();
            navigateFrame(activeViewport, 1);
            break;
        }
      });

      // Test function to verify frame validation works (for debugging)
      function testFrameValidation() {
        console.log('Testing frame validation...');
        
        // Test with valid frame number
        validateFrameInput(1, '5');
        console.log('âœ“ Valid frame test completed');
        
        // Test with out-of-range frame number
        validateFrameInput(1, '999');
        console.log('âœ“ Out-of-range frame test completed');
        
        // Test with non-numeric input
        validateFrameInput(1, 'abc');
        console.log('âœ“ Non-numeric input test completed');
        
        console.log('Frame validation tests completed. Check for notifications.');
      }

      // Ensure frame input event listeners are properly attached
      function setupFrameInputListeners() {
        const frameInput1 = document.getElementById('frameInput1');
        const frameInput2 = document.getElementById('frameInput2');
        
        if (frameInput1) {
          // Remove existing listeners to avoid duplicates
          frameInput1.removeEventListener('keydown', frameInput1._enterHandler);
          frameInput1._enterHandler = function(event) {
            if (event.key === 'Enter') {
              event.preventDefault();
              goToFrame(1);
            }
          };
          frameInput1.addEventListener('keydown', frameInput1._enterHandler);
        }
        
        if (frameInput2) {
          // Remove existing listeners to avoid duplicates
          frameInput2.removeEventListener('keydown', frameInput2._enterHandler);
          frameInput2._enterHandler = function(event) {
            if (event.key === 'Enter') {
              event.preventDefault();
              goToFrame(2);
            }
          };
          frameInput2.addEventListener('keydown', frameInput2._enterHandler);
        }
      }

      // Setup listeners when DOM is ready
      document.addEventListener('DOMContentLoaded', setupFrameInputListeners);

      // Get the currently active viewport (viewport with focus or last interacted)
      function getActiveViewport() {
        // Check if any viewport has an image loaded
        const viewport1 = document.getElementById('viewportImage1');
        const viewport2 = document.getElementById('viewportImage2');
        
        if (viewport1 && viewport1.style.display !== 'none' && viewportData[1]) {
          return 1;
        } else if (viewport2 && viewport2.style.display !== 'none' && viewportData[2]) {
          return 2;
        }
        
        return null;
      }

      // Navigate frame using + and - buttons
      async function navigateFrame(viewportNumber, direction) {
        const frameSlider = document.getElementById(`frameSlider${viewportNumber}`);
        const data = viewportData[viewportNumber];
        
        if (!frameSlider || !data) {
          console.error(`No frame slider or data found for viewport ${viewportNumber}`);
          return;
        }

        const currentFrame = parseInt(frameSlider.value);
        const maxFrame = parseInt(frameSlider.max);
        let newFrame = currentFrame + direction;

        console.log(`Navigating frame: current=${currentFrame}, direction=${direction}, new=${newFrame}, max=${maxFrame}`);

        // Handle wrap-around
        if (newFrame < 0) {
          newFrame = maxFrame;
        } else if (newFrame > maxFrame) {
          newFrame = 0;
        }

        console.log(`After wrap-around: newFrame=${newFrame}`);

        // Temporarily disable slider events to prevent conflicts
        const originalOnInput = frameSlider.oninput;
        frameSlider.oninput = null;

        // Update slider value
        frameSlider.value = newFrame;

        // Update UI immediately
        updateFrameInfo(viewportNumber, newFrame);
        updateNavigationButtons(viewportNumber, newFrame, maxFrame + 1);

        // Restore slider events
        frameSlider.oninput = originalOnInput;

        // Load the new frame
        try {
          if (data.currentFileType === 'oct') {
            // Flattened OCT mode - load flattened OCT frame
            const eye = data.currentEye || (viewportNumber === 1 ? 'left' : 'right');
            console.log(`Loading flattened OCT frame ${newFrame} for ${eye} eye`);
            await loadE2EOCTFrame(viewportNumber, eye, newFrame);
          } else if (data.currentFileType === 'original_oct') {
            // Flattened Original OCT mode - load flattened original OCT frame
            const eye = data.currentEye || (viewportNumber === 1 ? 'left' : 'right');
            console.log(`Loading flattened original OCT frame ${newFrame} for ${eye} eye`);
            await loadOriginalOCTFrame(viewportNumber, newFrame, data.dicom_file_path, eye);
          } else if (data.currentFileType === 'dicom') {
            // DICOM/SLO mode - load E2E eye image
            const eye = data.currentEye || (viewportNumber === 1 ? 'left' : 'right');
            console.log(`Loading DICOM frame ${newFrame} for ${eye} eye`);
            await loadE2EEyeImage(viewportNumber, newFrame, data.dicom_file_path, eye);
          } else if (data.currentFileType === 'original_oct') {
            // Legacy OCT mode - load flattened original OCT frame
            const eye = viewportNumber === 1 ? 'left' : 'right';
            console.log(`Loading flattened original OCT frame ${newFrame} for ${eye} eye`);
            await loadOriginalOCTFrame(viewportNumber, newFrame, data.dicom_file_path, eye);
          } else if (isE2EMode && data && data.dicom_file_path) {
            // Legacy E2E mode - load E2E eye image
            const eye = viewportNumber === 1 ? 'left' : 'right';
            console.log(`Loading E2E frame ${newFrame} for ${eye} eye`);
            await loadE2EEyeImage(viewportNumber, newFrame, data.dicom_file_path, eye);
          } else {
            // Regular mode - load regular frame
            console.log(`Loading regular frame ${newFrame}`);
            await loadFrame(viewportNumber, newFrame);
          }
        } catch (error) {
          console.error(`Error navigating frame: ${error}`);
          // Revert on error
          frameSlider.value = currentFrame;
          updateFrameInfo(viewportNumber, currentFrame);
          updateNavigationButtons(viewportNumber, currentFrame, maxFrame + 1);
        }
      }

      // Update frame info display
      function updateFrameInfo(viewportNumber, frameNumber) {
        const frameInfo = document.getElementById(`frameInfo${viewportNumber}`);
        if (frameInfo) {
          const eyeData = viewportNumber === 1 ? leftEyeFrameData : rightEyeFrameData;
          const data = viewportData[viewportNumber];
          let totalFramesCount;
          let modeText = '';
          
          if (data && data.currentFileType) {
            // Use the current file type to determine frame count
            const eye = data.currentEye || (viewportNumber === 1 ? 'left' : 'right');
            if (data.currentFileType === 'oct') {
              totalFramesCount = s3TreeData[`${eye}_eye`]?.oct?.length || 0;
              modeText = ' (Flattened OCT)';
            } else if (data.currentFileType === 'original_oct') {
              totalFramesCount = s3TreeData[`${eye}_eye`]?.original_oct?.length || 0;
              modeText = ' (Original OCT)';
            } else if (data.currentFileType === 'dicom') {
              totalFramesCount = s3TreeData[`${eye}_eye`]?.dicom?.length || 0;
              modeText = ' (DICOM)';
            } else {
              totalFramesCount = eyeData.totalFrames;
            }
          } else if (eyeData.isOctMode && eyeData.octFrameCount > 0) {
            // Legacy OCT mode
            totalFramesCount = eyeData.octFrameCount;
            modeText = ' (OCT)';
          } else {
            // Default fallback
            totalFramesCount = eyeData.totalFrames;
          }
          
          // Always show frame number as 1-based (frameNumber + 1) for user-friendly display
          frameInfo.textContent = `Frame ${frameNumber + 1} of ${totalFramesCount}${modeText}`;
        }

        // Also update the frame input field
        const frameInput = document.getElementById(`frameInput${viewportNumber}`);
        if (frameInput) {
          frameInput.value = frameNumber + 1; // Convert from 0-based to 1-based
        }
      }

      // Validate frame input in real-time and show warning if out of range
      function validateFrameInput(viewportNumber, inputValue) {
        const frameSlider = document.getElementById(`frameSlider${viewportNumber}`);
        const frameInput = document.getElementById(`frameInput${viewportNumber}`);
        const eyeData = viewportNumber === 1 ? leftEyeFrameData : rightEyeFrameData;
        const data = viewportData[viewportNumber];
        
        if (!frameSlider || !eyeData || !data || !frameInput) {
          return; // No validation if elements not found
        }
        
        // Parse input value
        const inputFrame = parseInt(inputValue, 10);
        if (isNaN(inputFrame)) {
          // Remove error styling for non-numeric input
          frameInput.classList.remove('error');
          return; // Don't show warning for non-numeric input
        }
        
        // Determine max frame (1-based for user)
        let maxFrame;
        if (eyeData.isOctMode && eyeData.octFrameCount > 0) {
          maxFrame = eyeData.octFrameCount;
        } else {
          maxFrame = parseInt(frameSlider.max, 10) + 1;
        }
        
        // Check if out of range
        if (inputFrame < 1 || inputFrame > maxFrame) {
          // Add error styling
          frameInput.classList.add('error');
          
          // Only show notification if the input is complete (not while typing)
          // This prevents spam notifications while user is still typing
          if (inputValue.length >= 2 || inputFrame > maxFrame) {
            showNotification(`Frame number ${inputFrame} is out of range. Please enter a value between 1 and ${maxFrame}.`, 'warning', 3000);
          }
        } else {
          // Remove error styling for valid input
          frameInput.classList.remove('error');
        }
      }

      // Go to specific frame number (1-based input, converted to 0-based for internal use)
      async function goToFrame(viewportNumber) {
        const frameInput = document.getElementById(`frameInput${viewportNumber}`);
        const frameSlider = document.getElementById(`frameSlider${viewportNumber}`);
        const eyeData = viewportNumber === 1 ? leftEyeFrameData : rightEyeFrameData;
        const data = viewportData[viewportNumber];

        if (!frameInput || !frameSlider || !eyeData || !data) {
          console.error(`Frame input elements not found for viewport ${viewportNumber}`);
          return;
        }

        // Get the frame number from the input (1-based from user)
        let inputFrame = parseInt(frameInput.value, 10);
        if (isNaN(inputFrame) || inputFrame < 1) {
          console.error(`Invalid frame number: ${frameInput.value}`);
          return;
        }

        // Determine max frame (1-based for user)
        let maxFrame;
        if (eyeData.isOctMode && eyeData.octFrameCount > 0) {
          maxFrame = eyeData.octFrameCount;
        } else {
          maxFrame = parseInt(frameSlider.max, 10) + 1;
        }

        // Validate inputFrame (1-based)
        if (inputFrame < 1 || inputFrame > maxFrame) {
          console.error(`Frame number ${inputFrame} out of range (1-${maxFrame})`);
          // Show user-friendly popup warning
          showNotification(`Frame number ${inputFrame} is out of range. Please enter a value between 1 and ${maxFrame}.`, 'warning', 4000);
          // Add error styling to input field
          frameInput.classList.add('error');
          // Reset input to current frame (slider is 0-based, so +1 for display)
          frameInput.value = parseInt(frameSlider.value, 10) + 1;
          return;
        }

        // Remove error styling since input is valid
        frameInput.classList.remove('error');

        // Only convert from 1-based to 0-based here
        const frameNumber = inputFrame - 1;

        // Log for debugging
        console.log(`Going to frame ${inputFrame} (internal: ${frameNumber}) for viewport ${viewportNumber}`);

        // Temporarily disable slider events to prevent conflicts
        const originalOnInput = frameSlider.oninput;
        frameSlider.oninput = null;

        // Update slider value (0-based)
        frameSlider.value = frameNumber;

        // Update UI immediately (pass 0-based frameNumber)
        updateFrameInfo(viewportNumber, frameNumber);
        updateNavigationButtons(viewportNumber, frameNumber, maxFrame);

        // Restore slider events
        frameSlider.oninput = originalOnInput;

        // Load the new frame (always pass 0-based frameNumber)
        try {
          if (data.currentFileType === 'oct') {
            // Flattened OCT mode - load flattened OCT frame
            const eye = data.currentEye || (viewportNumber === 1 ? 'left' : 'right');
            console.log(`Loading flattened OCT frame ${frameNumber} for ${eye} eye`);
            await loadE2EOCTFrame(viewportNumber, eye, frameNumber);
          } else if (data.currentFileType === 'original_oct') {
            // Flattened Original OCT mode - load flattened original OCT frame
            const eye = data.currentEye || (viewportNumber === 1 ? 'left' : 'right');
            console.log(`Loading flattened original OCT frame ${frameNumber} for ${eye} eye`);
            await loadOriginalOCTFrame(viewportNumber, frameNumber, data.dicom_file_path, eye);
          } else if (data.currentFileType === 'dicom') {
            // DICOM/SLO mode - load E2E eye image
            const eye = data.currentEye || (viewportNumber === 1 ? 'left' : 'right');
            console.log(`Loading DICOM frame ${frameNumber} for ${eye} eye`);
            await loadE2EEyeImage(viewportNumber, frameNumber, data.dicom_file_path, eye);
          } else if (eyeData.isOctMode && eyeData.octFrameCount > 0) {
            // Legacy OCT mode - load original OCT frame
            const eye = viewportNumber === 1 ? 'left' : 'right';
            console.log(`Loading OCT frame ${frameNumber} for ${eye} eye`);
            await loadOriginalOCTFrame(viewportNumber, frameNumber, data.dicom_file_path, eye);
          } else if (isE2EMode && data && data.dicom_file_path) {
            // Legacy E2E mode - load E2E eye image
            const eye = viewportNumber === 1 ? 'left' : 'right';
            console.log(`Loading E2E frame ${frameNumber} for ${eye} eye`);
            await loadE2EEyeImage(viewportNumber, frameNumber, data.dicom_file_path, eye);
          } else {
            // Regular mode - load regular frame
            console.log(`Loading regular frame ${frameNumber}`);
            await loadFrame(viewportNumber, frameNumber);
          }
        } catch (error) {
          console.error(`Error going to frame: ${error}`);
          // Revert on error
          frameSlider.value = parseInt(frameSlider.value, 10);
          updateFrameInfo(viewportNumber, parseInt(frameSlider.value, 10));
          updateNavigationButtons(viewportNumber, parseInt(frameSlider.value, 10), maxFrame);
        }
      }

      // Sync slider state with current frame data
      function syncSliderState(viewportNumber) {
        const frameSlider = document.getElementById(`frameSlider${viewportNumber}`);
        const eyeData = viewportNumber === 1 ? leftEyeFrameData : rightEyeFrameData;
        const data = viewportData[viewportNumber];
        
        if (!frameSlider || !eyeData) return;
        
        // Determine current frame count based on file type
        let numFrames;
        if (data && data.currentFileType) {
          const eye = data.currentEye || (viewportNumber === 1 ? 'left' : 'right');
          if (data.currentFileType === 'oct') {
            numFrames = s3TreeData[`${eye}_eye`]?.oct?.length || 0;
          } else if (data.currentFileType === 'original_oct') {
            numFrames = s3TreeData[`${eye}_eye`]?.original_oct?.length || 0;
          } else if (data.currentFileType === 'dicom') {
            numFrames = s3TreeData[`${eye}_eye`]?.dicom?.length || 0;
          } else {
            numFrames = eyeData.totalFrames;
          }
        } else if (eyeData.isOctMode && eyeData.octFrameCount > 0) {
          // Legacy OCT mode
          numFrames = eyeData.octFrameCount;
        } else {
          // Default fallback
          numFrames = eyeData.totalFrames;
        }
        
        // Update slider range if needed
        if (parseInt(frameSlider.max) !== numFrames - 1) {
          frameSlider.max = Math.max(0, numFrames - 1);
        }
        
        // Ensure current value is within range
        const currentValue = parseInt(frameSlider.value) || 0;
        if (currentValue < 0 || currentValue > numFrames - 1) {
          frameSlider.value = 0;
        }
        
        // Update UI
        updateFrameInfo(viewportNumber, parseInt(frameSlider.value) || 0);
        updateNavigationButtons(viewportNumber, parseInt(frameSlider.value) || 0, numFrames);
      }

      // Enhanced flatten function that caches only the flattened result for OCT images
      async function flattenImageDirectly(viewportNumber) {
        const data = viewportData[viewportNumber];
        if (!data) {
          throw new Error("No DICOM data loaded in this viewport.");
        }

        const img = document.getElementById(`viewportImage${viewportNumber}`);

        try {
          // Generate cache key for flattened OCT image
          const flattenMetadata = {
            path: data.s3_key,
            flattened: true,
            frame: 0,
          };

          const flattenCacheKey = imageCache.generateCacheKey(
            data.s3_key + "_flattened",
            flattenMetadata,
          );

          // Check if flattened version is already cached
          const cachedFlattened = imageCache.get(flattenCacheKey);
          if (cachedFlattened) {
            console.log("Using cached flattened OCT image");
            img.onload = () => {
              console.log("Cached flattened OCT image loaded successfully");
              img.style.display = "block";
              resetZoom(viewportNumber);
              setupImageInteractions(viewportNumber);
            };
            img.src = cachedFlattened.imageData;
            return;
          }

          console.log("Flattening OCT image and caching result...");
          const flattenResponse = await fetch(
            `/api/flatten_dicom_image?dicom_file_path=${encodeURIComponent(data.dicom_file_path)}`,
          );

          if (!flattenResponse.ok) {
            const errorData = await flattenResponse.json().catch(() => ({}));
            throw new Error(
              errorData.error ||
                `OCT flattening failed: ${flattenResponse.statusText}`,
            );
          }

          const flattenedBlob = await flattenResponse.blob();
          const flattenedUrl = URL.createObjectURL(flattenedBlob);

          // Cache ONLY the flattened image for OCT
          const cacheMetadata = {
            path: data.s3_key,
            flattened: true,
            size: flattenedBlob.size,
            lastModified: Date.now(),
            contentType: "image/png",
          };

          imageCache.set(flattenCacheKey, flattenedUrl, cacheMetadata);
          console.log(
            `Flattened OCT image cached with CRC: ${flattenCacheKey.substring(0, 8)}`,
          );

          // Store flattened URL in viewport data
          viewportData[viewportNumber].flattenedUrl = flattenedUrl;
          viewportData[viewportNumber].isFlattened = true;

          img.onload = () => {
            console.log("Flattened OCT image loaded and cached successfully");
            img.style.display = "block";
            resetZoom(viewportNumber);
            setupImageInteractions(viewportNumber);
          };

          img.src = flattenedUrl;
        } catch (error) {
          console.error("Error flattening OCT image:", error);
          throw error;
        }
      }

      // Flatten and show flattened image (enhanced with caching)
      async function flattenImage(viewportNumber) {
        const data = viewportData[viewportNumber];
        if (!data) {
          alert("No DICOM data loaded in this viewport.");
          return;
        }

        const img = document.getElementById(`viewportImage${viewportNumber}`);

        try {
          // Check cache first
          if (data.flattenedUrl) {
            console.log("Using cached flattened image");
            img.src = data.flattenedUrl;
            return;
          }

          console.log("Flattening image...");
          const flattenResponse = await fetch(
            `/api/flatten_dicom_image?dicom_file_path=${encodeURIComponent(data.dicom_file_path)}`,
          );

          if (!flattenResponse.ok) {
            const errorData = await flattenResponse.json().catch(() => ({}));
            throw new Error(
              errorData.error ||
                `Flattening failed: ${flattenResponse.statusText}`,
            );
          }

          const flattenedBlob = await flattenResponse.blob();
          const flattenedUrl = URL.createObjectURL(flattenedBlob);

          // Cache flattened image URL
          viewportData[viewportNumber].flattenedUrl = flattenedUrl;

          // Also cache in CRC cache
          try {
            const flattenMetadata = {
              path: data.s3_key,
              flattened: true,
              size: flattenedBlob.size,
              lastModified: Date.now(),
            };

            const flattenCacheKey = imageCache.generateCacheKey(
              data.s3_key + "_flattened",
              flattenMetadata,
            );
            imageCache.set(flattenCacheKey, flattenedUrl, flattenMetadata);
            console.log(
              `Flattened image cached with CRC: ${flattenCacheKey.substring(0, 8)}`,
            );
          } catch (cacheError) {
            console.warn(
              `Failed to cache flattened image: ${cacheError.message}`,
            );
          }

          img.onload = () => {
            console.log("Flattened image loaded successfully");
          };

          img.src = flattenedUrl;
        } catch (error) {
          console.error("Error flattening image:", error);
          alert(`Error flattening image: ${error.message}`);
        }
      }

      // Setup image interactions (zoom, pan)
      function setupImageInteractions(viewportNumber) {
        const img = document.getElementById(`viewportImage${viewportNumber}`);
        const container = document.getElementById(
          `viewportContent${viewportNumber}`,
        );

        if (!img || !container) return;

        container.onwheel = null;
        img.onmousedown = null;
        img.onmouseup = null;
        img.onmouseleave = null;
        img.onmousemove = null;

        container.onwheel = (e) => {
          e.preventDefault();
          const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
          zoomViewport(viewportNumber, zoomFactor);
        };

        img.onmousedown = (e) => {
          isDragging[viewportNumber] = true;
          lastMousePos[viewportNumber] = { x: e.clientX, y: e.clientY };
          img.style.cursor = 'grabbing';
          e.preventDefault();
        };
        img.onmouseup = () => {
          isDragging[viewportNumber] = false;
          img.style.cursor = 'grab';
        };
        img.onmouseleave = () => {
          isDragging[viewportNumber] = false;
          img.style.cursor = 'grab';
        };
        img.onmousemove = (e) => {
          if (isDragging[viewportNumber]) {
            const deltaX = e.clientX - lastMousePos[viewportNumber].x;
            const deltaY = e.clientY - lastMousePos[viewportNumber].y;
            lastMousePos[viewportNumber] = { x: e.clientX, y: e.clientY };
            viewportPan[viewportNumber].x += deltaX;
            viewportPan[viewportNumber].y += deltaY;
            updateImageTransform(viewportNumber);
          }
        };
        img.style.cursor = 'grab';
      }

      // Global mouse event handlers
      document.addEventListener("mousemove", (e) => {
        for (let viewportNumber of [1, 2]) {
          if (isDragging[viewportNumber]) {
            const deltaX = e.clientX - lastMousePos[viewportNumber].x;
            const deltaY = e.clientY - lastMousePos[viewportNumber].y;

            viewportPan[viewportNumber].x += deltaX;
            viewportPan[viewportNumber].y += deltaY;

            updateImageTransform(viewportNumber);

            lastMousePos[viewportNumber] = { x: e.clientX, y: e.clientY };
          }
        }
      });

      document.addEventListener("mouseup", () => {
        for (let viewportNumber of [1, 2]) {
          if (isDragging[viewportNumber]) {
            isDragging[viewportNumber] = false;
            const img = document.getElementById(
              `viewportImage${viewportNumber}`,
            );
            // if (img) img.style.cursor = "grab"; // Remove setting grab cursor
          }
        }
      });

      // Zoom viewport
      function zoomViewport(viewportNumber, factor) {
        const oldZoom = viewportZoom[viewportNumber];
        viewportZoom[viewportNumber] *= factor;
        viewportZoom[viewportNumber] = Math.max(
          0.1,
          Math.min(20, viewportZoom[viewportNumber]), // Increased max zoom from 5x (500%) to 20x (2000%)
        );

        // Adjust pan to maintain center point during zoom
        const zoomRatio = viewportZoom[viewportNumber] / oldZoom;
        viewportPan[viewportNumber].x *= zoomRatio;
        viewportPan[viewportNumber].y *= zoomRatio;

        updateImageTransform(viewportNumber);
        updateZoomDisplay(viewportNumber);
      }

      // Reset zoom and pan
      function resetZoom(viewportNumber) {
        viewportZoom[viewportNumber] = 1;
        viewportPan[viewportNumber] = { x: 0, y: 0 };
        updateImageTransform(viewportNumber);
        updateZoomDisplay(viewportNumber);
      }

      // Center image in viewport
      function centerImage(viewportNumber) {
        const img = document.getElementById(`viewportImage${viewportNumber}`);
        const container = document.getElementById(`viewportContent${viewportNumber}`);
        
        if (!img || !container) return;
        
        // Reset pan to center the image
        viewportPan[viewportNumber] = { x: 0, y: 0 };
        updateImageTransform(viewportNumber);
      }

      // Update image transform with improved constraints
      function constrainPan(viewportNumber) {
        const img = document.getElementById(`viewportImage${viewportNumber}`);
        const container = document.getElementById(
          `viewportContent${viewportNumber}`,
        );

        if (!img || !container) return;

        const zoom = viewportZoom[viewportNumber];
        
        // Get container dimensions
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        
        // Get original image dimensions
        const imgWidth = img.naturalWidth || img.width;
        const imgHeight = img.naturalHeight || img.height;
        
        // Calculate scaled image dimensions
        const scaledWidth = imgWidth * zoom;
        const scaledHeight = imgHeight * zoom;
        
        // Calculate maximum allowed pan values
        // When image is smaller than container, no pan is allowed
        // When image is larger than container, pan is limited to keep image within bounds
        const maxPanX = Math.max(0, (scaledWidth - containerWidth) / 2);
        const maxPanY = Math.max(0, (scaledHeight - containerHeight) / 2);
        
        // Constrain pan values to keep image within viewport
        viewportPan[viewportNumber].x = Math.max(
          -maxPanX,
          Math.min(maxPanX, viewportPan[viewportNumber].x),
        );
        viewportPan[viewportNumber].y = Math.max(
          -maxPanY,
          Math.min(maxPanY, viewportPan[viewportNumber].y),
        );
        
        // Additional constraint: if image is smaller than container, center it
        if (scaledWidth <= containerWidth) {
          viewportPan[viewportNumber].x = 0;
        }
        if (scaledHeight <= containerHeight) {
          viewportPan[viewportNumber].y = 0;
        }
      }

      function updateImageTransform(viewportNumber) {
        const img = document.getElementById(`viewportImage${viewportNumber}`);
        const container = document.getElementById(`viewportContent${viewportNumber}`);
        if (!img || !container) return;

        // Apply constraints before updating transform
        constrainPan(viewportNumber);

        const zoom = viewportZoom[viewportNumber];
        const pan = viewportPan[viewportNumber];

        // Apply transform with proper centering
        img.style.transform = `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`;
        
        // Ensure image stays within container bounds
        img.style.maxWidth = "100%";
        img.style.maxHeight = "100%";
        img.style.objectFit = "contain";
      }

      // Update zoom display
      function updateZoomDisplay(viewportNumber) {
        const zoomDisplay = document.getElementById(
          `zoomLevel${viewportNumber}`,
        );
        if (zoomDisplay) {
          zoomDisplay.textContent = `${Math.round(viewportZoom[viewportNumber] * 100)}%`;
        }
      }

      function updateS3TreeProgress(percent) {
        const container = document.getElementById("fileTreeContainer");
        container.innerHTML = `
    <div class="tree-loading">
      <div class="loader"></div>
      <span style="margin-left: 10px;">Loading... ${percent}%</span>
    </div>
  `;
      }

      // Enhanced switchToOCTMode function with separate eye data
      async function switchToOCTMode(viewportNumber) {
        const eyeData = viewportNumber === 1 ? leftEyeFrameData : rightEyeFrameData;
        const data = viewportData[viewportNumber];
        console.log(`Attempting to switch viewport ${viewportNumber} to OCT mode`);
        console.log(`Eye data:`, eyeData);
        console.log(`OCT frame count:`, eyeData?.octFrameCount);
        
        if (eyeData && eyeData.octFrameCount > 0) {
          eyeData.isOctMode = true;
          data.isOctMode = true;
          setupFrameSlider(viewportNumber);
          console.log(`Switched viewport ${viewportNumber} to OCT mode with ${eyeData.octFrameCount} frames`);
          
          // Go to the middle OCT frame and sync slider
          const eye = viewportNumber === 1 ? 'left' : 'right';
          const frameSlider = document.getElementById(`frameSlider${viewportNumber}`);
          const middleIndex = Math.floor(eyeData.octFrameCount / 2);
          try {
            if (frameSlider) {
              const originalOnInput = frameSlider.oninput;
              frameSlider.oninput = null;
              frameSlider.value = middleIndex;
              eyeData.currentFrame = middleIndex;
              updateFrameInfo(viewportNumber, middleIndex);
              updateNavigationButtons(viewportNumber, middleIndex, eyeData.octFrameCount);
              frameSlider.oninput = originalOnInput;
            }
            await loadOriginalOCTFrame(viewportNumber, middleIndex, data.dicom_file_path, eye);
          } catch (error) {
            console.error(`Error loading middle OCT frame: ${error}`);
          }
        } else {
          console.warn(`Cannot switch to OCT mode for viewport ${viewportNumber}: no OCT frames available`);
          console.warn(`Eye data available:`, eyeData);
        }
      }

      // Enhanced switchToRegularMode function with separate eye data
      async function switchToRegularMode(viewportNumber) {
        const eyeData = viewportNumber === 1 ? leftEyeFrameData : rightEyeFrameData;
        const data = viewportData[viewportNumber];
        if (eyeData && data) {
          // Preserve OCT frame count and data when switching to regular mode
          const octFrameCount = eyeData.octFrameCount;
          
          eyeData.isOctMode = false;
          data.isOctMode = false;
          // Keep the OCT frame count available for switching back
          eyeData.octFrameCount = octFrameCount;
          data.octFrameCount = octFrameCount;
          
          setupFrameSlider(viewportNumber);
          console.log(`Switched viewport ${viewportNumber} to regular mode, preserved ${octFrameCount} OCT frames for switching back`);
          
          // Load the first regular frame
          const eye = viewportNumber === 1 ? 'left' : 'right';
          try {
            await loadE2EEyeImage(viewportNumber, 0, data.dicom_file_path, eye);
          } catch (error) {
            console.error(`Error loading first regular frame: ${error}`);
          }
        }
      }

      // Enhanced loadOriginalOCTFrame function with separate eye data
      async function loadOriginalOCTFrame(
        viewportNumber,
        frameIndex,
        dicomFilePath,
        eye,
      ) {
        const img = document.getElementById(`viewportImage${viewportNumber}`);
        const placeholder = document.querySelector(
          `#viewportContent${viewportNumber} .viewport-placeholder`,
        );

        try {
          console.log(`Loading flattened original OCT frame ${frameIndex} for ${eye} eye from ${dicomFilePath}`);
          console.log(`Current eye data:`, viewportNumber === 1 ? leftEyeFrameData : rightEyeFrameData);
          
          updateProgress(
            viewportNumber,
            50,
            `Loading flattened original OCT frame ${frameIndex + 1}...`,
          );

          // Use the new flattened original OCT frame endpoint
          const url = `/api/view_e2e_oct_frame?dicom_file_path=${encodeURIComponent(dicomFilePath)}&eye=${eye}&frame_idx=${frameIndex}`;
          console.log(`Fetching: ${url}`);
          
          const response = await fetch(url);

          if (!response.ok) {
            const errorText = await response.text();
            console.error(`Flattened original OCT frame endpoint error: ${response.status} ${response.statusText}`);
            console.error(`Error details: ${errorText}`);
            throw new Error(
              `Failed to get flattened original OCT frame: ${response.statusText}`,
            );
          }

          const imageBlob = await response.blob();
          const imageUrl = URL.createObjectURL(imageBlob);

          img.onload = () => {
            console.log(`Flattened original OCT frame ${frameIndex} loaded for ${eye} eye`);
            img.style.display = "block";
            if (placeholder) placeholder.style.display = "none";
            resetZoom(viewportNumber);
            setupImageInteractions(viewportNumber);
            updateFrameInfo(viewportNumber, frameIndex);
            updateProgress(
              viewportNumber,
              100,
              `Flattened original OCT frame loaded successfully`,
            );
          };

          img.onerror = () => {
            throw new Error(`Failed to load flattened original OCT frame`);
          };

          img.src = imageUrl;
          
          // Update eye data current frame
          const eyeData = viewportNumber === 1 ? leftEyeFrameData : rightEyeFrameData;
          eyeData.currentFrame = frameIndex;
          
          // Set OCT mode and update frame count
          if (viewportData[viewportNumber]) {
            viewportData[viewportNumber].isOctMode = true;
            // Get the total number of flattened original OCT frames for this eye
            eyeData.isOctMode = true;
            
            // Update total frames for flattened original OCT
            const originalOCTCount = s3TreeData[`${eye}_eye`]?.original_oct?.length || 0;
            eyeData.totalFrames = originalOCTCount;
            
            // Store file type in viewport data for navigation
            viewportData[viewportNumber].currentFileType = 'original_oct';
            viewportData[viewportNumber].currentEye = eye;
            viewportData[viewportNumber].dicom_file_path = dicomFilePath;
            
            // Don't call setupFrameSlider here as it resets the slider value
            // The slider should already be set up correctly
          }
        } catch (error) {
          console.error(`Error loading flattened original OCT frame:`, error);
          throw error;
        }
      }

      // Enhanced E2E loading with better user feedback
      async function loadE2EEyeImage(
        viewportNumber,
        frameNumber,
        dicomFilePath,
        eye,
      ) {
        const img = document.getElementById(`viewportImage${viewportNumber}`);
        const placeholder = document.querySelector(
          `#viewportContent${viewportNumber} .viewport-placeholder`,
        );

        try {
          console.log(`Loading ${eye} eye frame ${frameNumber} from ${dicomFilePath}`);
          
          // Show enhanced loading feedback
          updateProgress(
            viewportNumber,
            30,
            `Loading ${eye} eye frame ${frameNumber + 1}...`,
            {
              eye: eye,
              frame: frameNumber + 1,
              status: "Loading"
            }
          );

          // Add visual feedback to the tree item being loaded
          highlightTreeItem(eye, frameNumber);

          // Use the existing E2E eye endpoint
          const url = `/api/view_e2e_eye?frame=${frameNumber}&dicom_file_path=${encodeURIComponent(dicomFilePath)}&eye=${eye}`;
          console.log(`Fetching: ${url}`);
          
          const response = await fetch(url);

          if (!response.ok) {
            const errorText = await response.text();
            console.error(`E2E eye endpoint error: ${response.status} ${response.statusText}`);
            console.error(`Error details: ${errorText}`);
            throw new Error(
              `Failed to get ${eye} eye image: ${response.statusText}`,
            );
          }

          updateProgress(
            viewportNumber,
            70,
            `Processing ${eye} eye image...`,
            {
              eye: eye,
              frame: frameNumber + 1,
              status: "Processing"
            }
          );

          const imageBlob = await response.blob();
          const imageUrl = URL.createObjectURL(imageBlob);

          img.onload = () => {
            console.log(`${eye} eye image loaded for frame ${frameNumber}`);
            img.style.display = "block";
            if (placeholder) placeholder.style.display = "none";
            resetZoom(viewportNumber);
            setupImageInteractions(viewportNumber);
            updateFrameInfo(viewportNumber, frameNumber);
            
            // Update tree item as loaded
            markTreeItemAsLoaded(eye, frameNumber);
            
            updateProgress(
              viewportNumber,
              100,
              `${eye} eye loaded successfully`,
              {
                eye: eye,
                frame: frameNumber + 1,
                status: "Complete"
              }
            );
          };

          img.onerror = () => {
            throw new Error(`Failed to load ${eye} eye image`);
          };

          img.src = imageUrl;
          
          // Update eye data current frame
          const eyeData = viewportNumber === 1 ? leftEyeFrameData : rightEyeFrameData;
          eyeData.currentFrame = frameNumber;
          
          // Set regular mode but preserve OCT frame data for switching back
          if (viewportData[viewportNumber]) {
            viewportData[viewportNumber].isOctMode = false;
            eyeData.isOctMode = false;
            
            // Update total frames for DICOM
            const dicomFrameCount = s3TreeData[`${eye}_eye`]?.dicom?.length || 0;
            eyeData.totalFrames = dicomFrameCount;
            
            // Store file type in viewport data for navigation
            viewportData[viewportNumber].currentFileType = 'dicom';
            viewportData[viewportNumber].currentEye = eye;
            viewportData[viewportNumber].dicom_file_path = dicomFilePath;
            
            // Don't clear octFrameCount - preserve it for switching back to OCT mode
            console.log(`Loaded regular frame ${frameNumber}, preserved ${eyeData.octFrameCount} OCT frames for switching back`);
          }
        } catch (error) {
          console.error(`Error loading ${eye} eye image:`, error);
          
          // Check if this is a "Not Found" error for missing eye data
          if (error.message && error.message.includes("Not Found")) {
            console.warn(`${eye} eye data not found - this is expected for single-eye E2E files`);
            
            // Update progress to indicate the eye is not available
            updateProgress(
              viewportNumber,
              100,
              `${eye} eye not available in this E2E file`,
            );
            
            // Don't throw the error - let the calling function handle it gracefully
            return false; // Indicate failure without throwing
          }
          
          // For other errors, still throw them
          throw error;
        }
      }

      // E2E File Loading Function - called from context menu
      async function loadE2EFile() {
        console.log("loadE2EFile called with selectedFilePath:", selectedFilePath);
        
        if (
                !selectedFilePath ||
                !selectedFilePath.toLowerCase().endsWith(".e2e")
            ) {
                console.error("Invalid file selected:", selectedFilePath);
                alert("Please select a valid E2E file");
                return;
            }

            console.log("Clearing viewports...");
            // Clear both viewports before loading E2E file
            clearViewport(1);
            clearViewport(2);

            console.log("Showing progress bars...");
            // Show progress bars for both viewports right away
            showProgress(1, "Preparing to load E2E...", {
                fileName: selectedFilePath.split("/").pop(),
                fileType: "E2E",
                Type: "E2E",
                eye: "Left",
                Eye: "Left",
            });
            showProgress(2, "Preparing to load E2E...", {
                fileName: selectedFilePath.split("/").pop(),
                fileType: "E2E",
                Type: "E2E",
                eye: "Right",
                Eye: "Right",
            });

            console.log("Hiding context menu...");
            document.getElementById("contextMenu").style.display = "none";
        // Reset E2E mode if already in it
        try {
          // Initialize E2E mode with enhanced user experience
          initializeE2EMode(selectedFilePath);

          // Download and process the E2E file
          const operationId1 = showProgress(1, "Processing E2E file...", {
            fileName: selectedFilePath.split("/").pop(),
            fileType: "E2E",
            Type: "E2E",
            eye: "Left",
            Eye: "Left",
          });

          const operationId2 = showProgress(2, "Processing E2E file...", {
            fileName: selectedFilePath.split("/").pop(),
            fileType: "E2E",
            Type: "E2E",
            eye: "Right",
            Eye: "Right",
          });

          // Generate operation ID for progress tracking
          const operationId = `download_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
          
          // Create abort controller for this operation
          const abortController = new AbortController();
          
          console.log("Making API request to download E2E file...");
          const response = await fetch(
            `/api/download_dicom_from_s3?path=${encodeURIComponent(selectedFilePath)}&operation_id=${operationId}`,
          );

          console.log("API response status:", response.status, response.statusText);
          if (!response.ok) {
            console.error("API request failed:", response.status, response.statusText);
            throw new Error(
              `Failed to download E2E file: ${response.statusText}`,
            );
          }

          const e2eData = await response.json();
          

          
          console.log("E2E data received:", e2eData);
          console.log("Left eye data:", e2eData.left_eye_data);
          console.log("Right eye data:", e2eData.right_eye_data);

          // Initialize separate eye frame data
          leftEyeFrameData.eyeData = e2eData.left_eye_data;
          leftEyeFrameData.totalFrames = e2eData.left_eye_data?.dicom?.length || 1; // Only count DICOM frames for SLO
          leftEyeFrameData.octFrameCount = e2eData.left_eye_data?.original_oct?.length || 0;
          leftEyeFrameData.currentFrame = 0;
          leftEyeFrameData.isOctMode = false;
          
          rightEyeFrameData.eyeData = e2eData.right_eye_data;
          rightEyeFrameData.totalFrames = e2eData.right_eye_data?.dicom?.length || 1; // Only count DICOM frames for SLO
          rightEyeFrameData.octFrameCount = e2eData.right_eye_data?.original_oct?.length || 0;
          rightEyeFrameData.currentFrame = 0;
          rightEyeFrameData.isOctMode = false;

          // Store E2E data for both viewports with separate eye data
          viewportData[1] = {
            ...e2eData,
            eye: "left",
            s3_key: selectedFilePath,
            left_eye_data: e2eData.left_eye_data,
            octFrameCount: leftEyeFrameData.octFrameCount,
            isOctMode: false
          };
          viewportData[2] = {
            ...e2eData,
            eye: "right",
            s3_key: selectedFilePath,
            right_eye_data: e2eData.right_eye_data,
            octFrameCount: rightEyeFrameData.octFrameCount,
            isOctMode: false
          };

          // Get tree data and populate eye trees
          await populateE2ETreeData(e2eData.dicom_file_path);

          // Check if we have any frames before trying to load
          const leftFrames = leftEyeFrameData.totalFrames;
          const rightFrames = rightEyeFrameData.totalFrames;
          
          console.log(`Left frames: ${leftFrames}, Right frames: ${rightFrames}`);

          // Try to load both eyes and handle missing eye data gracefully
          console.log("Attempting to load both eyes...");
          
          let leftEyeLoaded = false;
          let rightEyeLoaded = false;
          
          // Try to load left eye
          if (leftFrames > 0) {
            try {
              leftEyeLoaded = await loadE2EEyeImage(1, 0, e2eData.dicom_file_path, "left");
            } catch (error) {
              console.warn("Failed to load left eye:", error.message);
              leftEyeLoaded = false;
            }
          }
          
          // Try to load right eye
          if (rightFrames > 0) {
            try {
              rightEyeLoaded = await loadE2EEyeImage(2, 0, e2eData.dicom_file_path, "right");
            } catch (error) {
              console.warn("Failed to load right eye:", error.message);
              rightEyeLoaded = false;
            }
          }
          
          // Implement fallback logic based on what actually loaded
          if (leftEyeLoaded && rightEyeLoaded) {
            // Both eyes loaded successfully
            console.log("Both eyes loaded successfully");
            updateViewportTitle(1, "Viewport 1 - Left Eye");
            updateViewportTitle(2, "Viewport 2 - Right Eye");
          } else if (leftEyeLoaded && !rightEyeLoaded) {
            // Only left eye loaded - show only in left viewport
            console.log("Only left eye loaded - showing in left viewport only");
            updateViewportTitle(1, "Viewport 1 - Left Eye");
            updateViewportTitle(2, "Viewport 2 - No Data");
            updateProgress(2, 100, "Right eye not available");
          } else if (!leftEyeLoaded && rightEyeLoaded) {
            // Only right eye loaded - show only in right viewport
            console.log("Only right eye loaded - showing in right viewport only");
            updateViewportTitle(1, "Viewport 1 - No Data");
            updateViewportTitle(2, "Viewport 2 - Right Eye");
            updateProgress(1, 100, "Left eye not available");
          } else {
            // Neither eye loaded
            console.warn("No eye data could be loaded");
            updateProgress(1, 100, "No eye data available");
            updateProgress(2, 100, "No eye data available");
          }

          // Setup frame sliders for both eyes using separate data
          setupFrameSlider(1);
          setupFrameSlider(2);

          // Hide progress
          hideProgress(1);
          hideProgress(2);

          console.log("E2E file loaded successfully");
          console.log("Left eye frame data:", leftEyeFrameData);
          console.log("Right eye frame data:", rightEyeFrameData);
        } catch (error) {
          console.error("Error loading E2E file:", error);
          hideProgress(1);
          hideProgress(2);
          alert(`Error loading E2E file: ${error.message}`);
          resetE2EMode();
        }
      }

      // Enhanced resetE2EMode function to clear separate eye data
      function resetE2EMode() {
        isE2EMode = false;
        currentE2EFile = null;
        focusedEye = null;

        // Reset separate eye frame data
        leftEyeFrameData.totalFrames = 0;
        leftEyeFrameData.currentFrame = 0;
        leftEyeFrameData.octFrameCount = 0;
        leftEyeFrameData.isOctMode = false;
        leftEyeFrameData.eyeData = null;
        
        rightEyeFrameData.totalFrames = 0;
        rightEyeFrameData.currentFrame = 0;
        rightEyeFrameData.octFrameCount = 0;
        rightEyeFrameData.isOctMode = false;
        rightEyeFrameData.eyeData = null;

        hideE2EControls();
        updateViewportTitles();
        resetEyeFocus();

        // Clear both viewports
        clearViewport(1);
        clearViewport(2);

        console.log("E2E mode reset");
      }

      // Helper function to update viewport titles for E2E mode
      function updateViewportTitles() {
        const title1 = document.querySelector("#viewport1 h4");
        const title2 = document.querySelector("#viewport2 h4");

        if (isE2EMode) {
          if (title1) title1.textContent = "Viewport 1 - Left Eye";
          if (title2) title2.textContent = "Viewport 2 - Right Eye";
        } else {
          if (title1) title1.textContent = "Viewport 1";
          if (title2) title2.textContent = "Viewport 2";
        }
      }

      // Helper function to update individual viewport title
      function updateViewportTitle(viewportNumber, title) {
        const titleElement = document.querySelector(`#viewport${viewportNumber} h4`);
        if (titleElement) {
          titleElement.textContent = title;
        }
      }

      // Helper function to show E2E controls
      function showE2EControls() {
        // Show the eye tree containers
        const leftEyeTreeContainer = document.getElementById('leftEyeTreeContainer');
        const rightEyeTreeContainer = document.getElementById('rightEyeTreeContainer');
        
        if (leftEyeTreeContainer) {
          leftEyeTreeContainer.style.display = 'block';
          // Expand the left eye tree by default
          const leftContent = document.getElementById('leftEyeTreeContent');
          const leftButton = leftEyeTreeContainer.querySelector('.tree-collapse-btn');
          if (leftContent && leftButton) {
            leftContent.style.display = 'block';
            leftEyeTreeContainer.classList.add('expanded');
            leftButton.classList.remove('collapsed');
            leftButton.innerHTML = '<i class="fas fa-chevron-up"></i>';
          }
        }
        
        if (rightEyeTreeContainer) {
          rightEyeTreeContainer.style.display = 'block';
          // Expand the right eye tree by default
          const rightContent = document.getElementById('rightEyeTreeContent');
          const rightButton = rightEyeTreeContainer.querySelector('.tree-collapse-btn');
          if (rightContent && rightButton) {
            rightContent.style.display = 'block';
            rightEyeTreeContainer.classList.add('expanded');
            rightButton.classList.remove('collapsed');
            rightButton.innerHTML = '<i class="fas fa-chevron-up"></i>';
          }
        }
        
        console.log("E2E controls and eye trees shown and expanded");
      }

      // Helper function to hide E2E controls
      function hideE2EControls() {
        // Hide the eye tree containers
        const leftEyeTreeContainer = document.getElementById('leftEyeTreeContainer');
        const rightEyeTreeContainer = document.getElementById('rightEyeTreeContainer');
        
        if (leftEyeTreeContainer) {
          leftEyeTreeContainer.style.display = 'none';
        }
        if (rightEyeTreeContainer) {
          rightEyeTreeContainer.style.display = 'none';
        }
        
        console.log("E2E controls and eye trees hidden");
      }

      // Helper functions for E2E tree visual feedback
      function highlightTreeItem(eye, frameNumber) {
        const treeContainer = document.getElementById(`${eye}EyeTreeContainer`);
        if (!treeContainer) return;

        // Remove previous highlights
        const previousHighlights = treeContainer.querySelectorAll('.tree-item-loading');
        previousHighlights.forEach(item => item.classList.remove('tree-item-loading'));

        // Find and highlight the current item
        const treeItems = treeContainer.querySelectorAll('.tree-item');
        if (treeItems[frameNumber]) {
          treeItems[frameNumber].classList.add('tree-item-loading');
        }
      }

      function markTreeItemAsLoaded(eye, frameNumber) {
        const treeContainer = document.getElementById(`${eye}EyeTreeContainer`);
        if (!treeContainer) return;

        // Remove loading highlight
        const loadingItem = treeContainer.querySelector('.tree-item-loading');
        if (loadingItem) {
          loadingItem.classList.remove('tree-item-loading');
          loadingItem.classList.add('tree-item-loaded');
        }
      }

      function clearTreeItemHighlights(eye) {
        const treeContainer = document.getElementById(`${eye}EyeTreeContainer`);
        if (!treeContainer) return;

        const items = treeContainer.querySelectorAll('.tree-item-loading, .tree-item-loaded');
        items.forEach(item => {
          item.classList.remove('tree-item-loading', 'tree-item-loaded');
        });
      }

      // Enhanced E2E mode initialization with better user guidance
      function initializeE2EMode(filePath) {
        console.log("Initializing E2E mode for:", filePath);
        
        // Show E2E mode indicator
        showE2EModeIndicator();
        
        // Switch to E2E mode
        isE2EMode = true;
        currentE2EFile = filePath;

        // Update viewport titles with clear indicators
        updateViewportTitles();

        // Show E2E controls with better styling
        showE2EControls();

        // Clear any previous highlights
        clearTreeItemHighlights('left');
        clearTreeItemHighlights('right');

        // Show user guidance
        showE2EUserGuidance();
      }

      function showE2EModeIndicator() {
        // E2E mode indicator removed as requested
      }

      function hideE2EModeIndicator() {
        // E2E mode indicator removed as requested
      }

      function showE2EUserGuidance() {
        // Show helpful guidance for E2E mode
        const leftTree = document.getElementById('leftEyeTreeContent');
        const rightTree = document.getElementById('rightEyeTreeContent');
        
        if (leftTree && leftTree.querySelector('.tree-placeholder')) {
          leftTree.innerHTML = `
            <div class="e2e-guidance">
              <div class="guidance-icon"><i class="fas fa-info-circle"></i></div>
              <div class="guidance-text">
                <strong>Left Eye Files</strong><br>
                Click on any file to load it in the left viewport
              </div>
            </div>
          `;
        }
        
        if (rightTree && rightTree.querySelector('.tree-placeholder')) {
          rightTree.innerHTML = `
            <div class="e2e-guidance">
              <div class="guidance-icon"><i class="fas fa-info-circle"></i></div>
              <div class="guidance-text">
                <strong>Right Eye Files</strong><br>
                Click on any file to load it in the right viewport
              </div>
            </div>
          `;
        }
      }

      // Helper function to reset eye focus
      function resetEyeFocus() {
        console.log('resetEyeFocus called');
        focusedEye = null;
        // Remove any visual focus indicators
        document.querySelectorAll('.eye-focused').forEach(el => {
          el.classList.remove('eye-focused');
        });
        
        // Show both viewports
        const viewport1 = document.getElementById('viewport1');
        const viewport2 = document.getElementById('viewport2');
        const viewportsContainer = document.getElementById('viewportsContainer');
        const viewportPanel = viewportsContainer ? viewportsContainer.closest('.viewport-panel') : null;
        
        if (viewport1) {
          viewport1.style.display = 'block';
          viewport1.style.visibility = 'visible';
          viewport1.style.width = '';
          viewport1.style.maxWidth = '';
          viewport1.style.flex = '';
          viewport1.style.minWidth = '';
          viewport1.style.gridColumn = '';
        }
        if (viewport2) {
          viewport2.style.display = 'block';
          viewport2.style.visibility = 'visible';
          viewport2.style.width = '';
          viewport2.style.maxWidth = '';
          viewport2.style.flex = '';
          viewport2.style.minWidth = '';
          viewport2.style.gridColumn = '';
        }
        
        // Remove focus mode class to restore normal layout
        if (viewportsContainer) {
          viewportsContainer.classList.remove('focus-mode');
          viewportsContainer.classList.add('side-by-side');
          // Explicitly restore two-column grid
          viewportsContainer.style.display = 'grid';
          viewportsContainer.style.gridTemplateColumns = '1fr 1fr';
          viewportsContainer.style.gridTemplateRows = '1fr';
          viewportsContainer.style.gap = '10px';
          viewportsContainer.style.width = '';
          viewportsContainer.style.maxWidth = '';
          viewportsContainer.style.flex = '';
          viewportsContainer.style.minWidth = '';
        }
        
        // Remove focus mode from viewport-panel
        if (viewportPanel) {
          viewportPanel.classList.remove('focus-mode');
          viewportPanel.style.flex = '';
          viewportPanel.style.width = '';
          viewportPanel.style.maxWidth = '';
          viewportPanel.style.minWidth = '';
          viewportPanel.style.flexGrow = '';
          viewportPanel.style.flexShrink = '';
          viewportPanel.style.flexBasis = '';
        }
        
        // Hide dropdown menu
        const dropdownMenu = document.getElementById('eyeFocusDropdownMenu');
        const burgerButton = document.getElementById('eyeFocusBurgerButton');
        if (dropdownMenu) dropdownMenu.classList.remove('show');
        if (burgerButton) burgerButton.classList.remove('active');
      }

      // Function to toggle the eye focus dropdown menu
      function toggleEyeFocusMenu() {
        const dropdownMenu = document.getElementById('eyeFocusDropdownMenu');
        const burgerButton = document.getElementById('eyeFocusBurgerButton');
        
        if (dropdownMenu && burgerButton) {
          const isVisible = dropdownMenu.classList.contains('show');
          
          if (isVisible) {
            // Hide dropdown
            dropdownMenu.classList.remove('show');
            burgerButton.classList.remove('active');
            burgerButton.setAttribute('aria-expanded', 'false');
          } else {
            // Show dropdown
            dropdownMenu.classList.add('show');
            burgerButton.classList.add('active');
            burgerButton.setAttribute('aria-expanded', 'true');
            // Ensure items are keyboard accessible
            dropdownMenu.querySelectorAll('.eye-focus-item').forEach((item) => {
              item.setAttribute('tabindex', '0');
              item.setAttribute('role', 'menuitem');
            });
          }
        }
      }

      // Function to focus on a specific eye (hide the other viewport)
      function focusOnEye(eye) {
        console.log('focusOnEye called with:', eye);
        const viewport1 = document.getElementById('viewport1');
        const viewport2 = document.getElementById('viewport2');
        const viewportsContainer = document.getElementById('viewportsContainer');
        const viewportPanel = viewportsContainer ? viewportsContainer.closest('.viewport-panel') : null;
        const dropdownMenu = document.getElementById('eyeFocusDropdownMenu');
        const burgerButton = document.getElementById('eyeFocusBurgerButton');
        
        if (!viewport1 || !viewport2 || !viewportsContainer) {
          console.error('Viewport elements not found!');
          return;
        }
        
        // Check if we're in side-by-side layout
        const isSideBySide = viewportsContainer.classList.contains('side-by-side');
        
        if (eye === 'left') {
          // Show left viewport (viewport1), hide right viewport (viewport2)
          console.log('Hiding right viewport, showing left viewport');
          // Hide right viewport first
          viewport2.style.display = 'none';
          viewport2.style.visibility = 'hidden';
          // Show and expand left viewport
          viewport1.style.display = 'block';
          viewport1.style.visibility = 'visible';
          viewport1.style.width = '100%';
          viewport1.style.maxWidth = '100%';
          viewport1.style.flex = '1 1 100%';
          viewport1.style.minWidth = '0';
          viewport1.style.gridColumn = '1 / -1';
          
          // Update container to single column
          viewportsContainer.classList.remove('side-by-side');
          viewportsContainer.classList.add('focus-mode');
          viewportsContainer.style.display = 'grid';
          viewportsContainer.style.gridTemplateColumns = '1fr';
          viewportsContainer.style.gridTemplateRows = '1fr';
          viewportsContainer.style.gap = '0px';
          viewportsContainer.style.width = '100%';
          viewportsContainer.style.maxWidth = '100%';
          viewportsContainer.style.flex = '1 1 100%';
          viewportsContainer.style.minWidth = '0';
          
          // Expand viewport-panel
          if (viewportPanel) {
            viewportPanel.classList.add('focus-mode');
            viewportPanel.style.flex = '1 1 auto';
            viewportPanel.style.width = '100%';
            viewportPanel.style.maxWidth = '100%';
            viewportPanel.style.minWidth = '0';
            viewportPanel.style.flexGrow = '1';
            viewportPanel.style.flexShrink = '1';
            viewportPanel.style.flexBasis = 'auto';
          }
          focusedEye = 'left';
        } else if (eye === 'right') {
          // Show right viewport (viewport2), hide left viewport (viewport1)
          console.log('Hiding left viewport, showing right viewport');
          // Hide left viewport first
          viewport1.style.display = 'none';
          viewport1.style.visibility = 'hidden';
          // Show and expand right viewport
          viewport2.style.display = 'block';
          viewport2.style.visibility = 'visible';
          viewport2.style.width = '100%';
          viewport2.style.maxWidth = '100%';
          viewport2.style.flex = '1 1 100%';
          viewport2.style.minWidth = '0';
          viewport2.style.gridColumn = '1 / -1';
          
          // Update container to single column
          viewportsContainer.classList.remove('side-by-side');
          viewportsContainer.classList.add('focus-mode');
          viewportsContainer.style.display = 'grid';
          viewportsContainer.style.gridTemplateColumns = '1fr';
          viewportsContainer.style.gridTemplateRows = '1fr';
          viewportsContainer.style.gap = '0px';
          viewportsContainer.style.width = '100%';
          viewportsContainer.style.maxWidth = '100%';
          viewportsContainer.style.flex = '1 1 100%';
          viewportsContainer.style.minWidth = '0';
          
          // Expand viewport-panel
          if (viewportPanel) {
            viewportPanel.classList.add('focus-mode');
            viewportPanel.style.flex = '1 1 auto';
            viewportPanel.style.width = '100%';
            viewportPanel.style.maxWidth = '100%';
            viewportPanel.style.minWidth = '0';
            viewportPanel.style.flexGrow = '1';
            viewportPanel.style.flexShrink = '1';
            viewportPanel.style.flexBasis = 'auto';
          }
          focusedEye = 'right';
        }
        
        // Hide dropdown menu after selection
        if (dropdownMenu) dropdownMenu.classList.remove('show');
        if (burgerButton) burgerButton.classList.remove('active');
        
        console.log('Focus mode applied. Viewport1 display:', viewport1.style.display, 'Viewport2 display:', viewport2.style.display);
      }

      // Expose functions to window for HTML onclick handlers
      window.focusOnEye = focusOnEye;
      window.resetEyeFocus = resetEyeFocus;
      window.toggleEyeFocusMenu = toggleEyeFocusMenu;

      // Helper function to populate E2E tree data
      async function populateE2ETreeData(dicomFilePath) {
        try {
          console.log(`Fetching E2E tree data for: ${dicomFilePath}`);
          const response = await fetch(`/api/get_e2e_tree_data?dicom_file_path=${encodeURIComponent(dicomFilePath)}`);
          if (response.ok) {
            const data = await response.json();
            console.log("E2E tree data populated:", data);
            console.log("Data keys:", Object.keys(data));
            console.log("Left eye data:", data.left_eye);
            console.log("Right eye data:", data.right_eye);
            
            // Check if the file is empty or corrupted
            const leftEyeTotal = (data.left_eye?.dicom?.length || 0) + 
                                (data.left_eye?.oct?.length || 0) + 
                                (data.left_eye?.original_oct?.length || 0);
            const rightEyeTotal = (data.right_eye?.dicom?.length || 0) + 
                                 (data.right_eye?.oct?.length || 0) + 
                                 (data.right_eye?.original_oct?.length || 0);
            
            if (leftEyeTotal === 0 && rightEyeTotal === 0) {
              console.warn("E2E file appears to be empty or corrupted - showing warning popup");
              showEmptyFileWarning();
              return;
            }
            
            // Store the tree data for later use
            s3TreeData = data;
            
            // Render the tree data in the eye tree containers
            renderE2ETreeData(data);
          } else {
            console.error("Failed to fetch E2E tree data:", response.status, response.statusText);
          }
        } catch (error) {
          console.error("Error fetching E2E tree data:", error);
        }
      }

      // Function to render E2E tree data in the eye tree containers
      function renderE2ETreeData(treeData) {
        console.log("Rendering E2E tree data:", treeData);
        
        // Render left eye tree
        if (treeData.left_eye) {
          renderEyeTree('left', treeData.left_eye);
        }
        
        // Render right eye tree
        if (treeData.right_eye) {
          renderEyeTree('right', treeData.right_eye);
        }
      }

      // Function to render individual eye tree
      function renderEyeTree(eye, eyeData) {
        const containerId = `${eye}EyeTreeContent`;
        const container = document.getElementById(containerId);
        
        console.log(`Rendering ${eye} eye tree. Container ID: ${containerId}`);
        console.log(`Container found:`, container);
        console.log(`Eye data:`, eyeData);
        
        if (!container) {
          console.error(`Container not found: ${containerId}`);
          return;
        }

        // Clear existing content
        container.innerHTML = '';
        
        // Create tree structure
        const treeHtml = createEyeTreeHTML(eye, eyeData);
        console.log(`Generated HTML for ${eye} eye:`, treeHtml);
        container.innerHTML = treeHtml;
        
        console.log(`${eye} eye tree rendered with data:`, eyeData);
      }

      // Function to create HTML for eye tree
      function createEyeTreeHTML(eye, eyeData) {
        console.log(`Creating HTML for ${eye} eye with data:`, eyeData);
        let html = '';
        
        // Fundus/SLO Images section (DICOM)
        if (eyeData.dicom && eyeData.dicom.length > 0) {
          console.log(`Found ${eyeData.dicom.length} Fundus/SLO images for ${eye} eye`);
          html += `
            <div class="eye-tree-folder">
              <div class="folder-header eye-tree-item" onclick="toggleFolder(this)">
                <i class="fas fa-chevron-right folder-arrow"></i>
                <i class="fas fa-camera folder-icon"></i>
                Fundus/SLO Images (${eyeData.dicom.length})
              </div>
              <div class="eye-tree-children" style="display: none;">
          `;
          
          eyeData.dicom.forEach((file, index) => {
            html += `
              <div class="eye-tree-item file-item fundus-image" onclick="selectEyeFile('${eye}', 'dicom', ${index})">
                <i class="fas fa-camera"></i>
                ${file.name || `Fundus ${index + 1}`}
              </div>
            `;
          });

          html += '</div></div>';
        } else {
          console.log(`No Fundus/SLO images found for ${eye} eye`);
        }
        
        // Flattened OCT Images section
        if (eyeData.oct && eyeData.oct.length > 0) {
          // Filter to only include flattened frames (defensive, in case of legacy data)
          const flattenedOCT = eyeData.oct.filter(file => {
            if (typeof file === 'string') {
              return file.includes('flattened');
            } else if (file && file.name) {
              return file.name.includes('flattened');
            }
            return false;
          });
          console.log(`Found ${flattenedOCT.length} Flattened OCT images for ${eye} eye`);
          if (flattenedOCT.length > 0) {
            html += `
              <div class="eye-tree-folder">
                <div class="folder-header eye-tree-item" onclick="toggleFolder(this)">
                  <i class="fas fa-chevron-right folder-arrow"></i>
                  <i class="fas fa-eye folder-icon"></i>
                  Flattened OCT Frames (${flattenedOCT.length})
                </div>
                <div class="eye-tree-children" style="display: none;">
            `;
            flattenedOCT.forEach((file, index) => {
              html += `
                <div class="eye-tree-item file-item flattened-oct" onclick="selectEyeFile('${eye}', 'oct', ${index})">
                  <i class="fas fa-eye"></i>
                  ${file.name || `Flattened OCT ${index + 1}`}
                </div>
              `;
            });
            html += '</div></div>';
          } else {
            console.log(`No Flattened OCT images found for ${eye} eye after filtering`);
          }
        } else {
          console.log(`No Flattened OCT images found for ${eye} eye`);
        }
        
        // Flattened Original OCT Frames section
        if (eyeData.original_oct && eyeData.original_oct.length > 0) {
                                      console.log(`Found ${eyeData.original_oct.length} Flattened Original OCT frames for ${eye} eye`);
          html += `
            <div class="eye-tree-folder">
              <div class="folder-header eye-tree-item" onclick="toggleFolder(this)">
                <i class="fas fa-chevron-right folder-arrow"></i>
                <i class="fas fa-layer-group folder-icon"></i>
                Flattened Original OCT Frames (${eyeData.original_oct.length})
              </div>
              <div class="eye-tree-children" style="display: none;">
          `;
          
          eyeData.original_oct.forEach((file, index) => {
            html += `
              <div class="eye-tree-item file-item original-oct-frame" onclick="selectEyeFile('${eye}', 'original_oct', ${index})">
                <i class="fas fa-layer-group"></i>
                ${file.name || `OCT Frame ${index + 1}`}
              </div>
            `;
          });

          html += '</div></div>';
        } else {
          console.log(`No Flattened Original OCT frames found for ${eye} eye`);
        }
        
        // If no files, show placeholder
        if ((!eyeData.dicom || eyeData.dicom.length === 0) && 
            (!eyeData.oct || eyeData.oct.length === 0) &&
            (!eyeData.original_oct || eyeData.original_oct.length === 0)) {
          console.log(`No files found for ${eye} eye, showing placeholder`);
          html = '<div class="tree-placeholder">No files available for this eye</div>';
        }
        
        console.log(`Generated HTML for ${eye} eye:`, html);
        return html;
      }

      // Function to toggle folder expansion
      function toggleFolder(folderHeader) {
        const folder = folderHeader.parentElement;
        const children = folder.querySelector('.eye-tree-children');
        const arrow = folderHeader.querySelector('.folder-arrow');
        
        if (children.style.display === 'none') {
          children.style.display = 'block';
          arrow.style.transform = 'rotate(90deg)';
          } else {
          children.style.display = 'none';
          arrow.style.transform = 'rotate(0deg)';
        }
      }

      // Function to select a file from the eye tree
      function selectEyeFile(eye, fileType, index) {
        console.log(`Selected ${fileType} file ${index} for ${eye} eye`);
        
        // Remove previous selection
        document.querySelectorAll('.eye-tree-item.selected').forEach(item => {
          item.classList.remove('selected');
        });
        
        // Add selection to clicked item
        event.target.closest('.eye-tree-item').classList.add('selected');
        
        // Load the selected file into the appropriate viewport
        const viewportNumber = eye === 'left' ? 1 : 2;
        
        // Get the file data from the tree data
        if (s3TreeData && s3TreeData[`${eye}_eye`]) {
          const eyeData = s3TreeData[`${eye}_eye`];
          const files = eyeData[fileType];
          
          if (files && files[index]) {
            const file = files[index];
            console.log(`Loading ${fileType} file:`, file);
            
            // Load the file into the viewport
            loadEyeFileIntoViewport(viewportNumber, eye, fileType, file);
          }
        }
      }

      // Function to load a file from the eye tree into a viewport
      async function loadEyeFileIntoViewport(viewportNumber, eye, fileType, file) {
        try {
          console.log(`Loading ${fileType} file into viewport ${viewportNumber}:`, file);

          // Store the current file type for navigation
          if (!viewportData[viewportNumber]) {
            viewportData[viewportNumber] = {};
          }
          viewportData[viewportNumber].currentFileType = fileType;
          viewportData[viewportNumber].currentEye = eye;
          viewportData[viewportNumber].dicom_file_path = currentE2EFile;

          // Update progress
          updateProgress(viewportNumber, 0, `Loading ${fileType} file...`);

          // Load the file based on type
          if (fileType === 'dicom') {
            // Load Fundus/SLO image
            await loadE2EEyeImage(viewportNumber, file.frame_index || 0, currentE2EFile, eye);
          } else if (fileType === 'oct') {
            // Load Flattened OCT image
            await loadE2EOCTFrame(viewportNumber, eye, file.frame_index || 0);
          } else if (fileType === 'original_oct') {
            // Load Flattened Original OCT frame
            await loadOriginalOCTFrame(viewportNumber, file.frame_index || 0, currentE2EFile, eye);
          }

          // --- Ensure frame slider and input are synced to the selected frame ---
          // This ensures that when an image is selected from the file tree, the slider/input reflect that frame
          const selectedIndex = file.frame_index || 0;
          const frameSlider = document.getElementById(`frameSlider${viewportNumber}`);
          const frameInput = document.getElementById(`frameInput${viewportNumber}`);
          if (frameSlider) {
            frameSlider.value = selectedIndex;
          }
          if (frameInput) {
            frameInput.value = selectedIndex + 1; // 1-based for user
          }
          updateFrameInfo(viewportNumber, selectedIndex);
          updateNavigationButtons(viewportNumber, selectedIndex, frameSlider ? parseInt(frameSlider.max) + 1 : 1);

          // Update frame slider if needed
          const frameData = eye === 'left' ? leftEyeFrameData : rightEyeFrameData;
          if (frameData.totalFrames > 0) {
            // For SLO images, update the frame slider with the correct frame count
            if (fileType === 'dicom') {
              const dicomFrameCount = s3TreeData[`${eye}_eye`]?.dicom?.length || 0;
              updateFrameSlider(viewportNumber, selectedIndex, dicomFrameCount);
            } else if (fileType === 'oct') {
              const flattenedOCTCount = s3TreeData[`${eye}_eye`]?.oct?.length || 0;
              updateFrameSlider(viewportNumber, selectedIndex, flattenedOCTCount);
            } else if (fileType === 'original_oct') {
              const originalOCTCount = s3TreeData[`${eye}_eye`]?.original_oct?.length || 0;
              updateFrameSlider(viewportNumber, selectedIndex, originalOCTCount);
            } else {
              updateFrameSlider(viewportNumber, selectedIndex, frameData.totalFrames);
            }
          }

          hideProgress(viewportNumber);

        } catch (error) {
          console.error(`Error loading ${fileType} file into viewport ${viewportNumber}:`, error);
          hideProgress(viewportNumber);
          showError(viewportNumber, `Error loading ${fileType} file: ${error.message}`);
        }
      }

      // Function to toggle eye tree visibility
      function toggleEyeTree(eye) {
        const container = document.getElementById(`${eye}EyeTreeContainer`);
        const content = document.getElementById(`${eye}EyeTreeContent`);
        const button = container.querySelector('.tree-collapse-btn');
        
        if (content.style.display === 'none') {
          content.style.display = 'block';
          container.classList.add('expanded');
          button.classList.remove('collapsed');
          button.innerHTML = '<i class="fas fa-chevron-up"></i>';
        } else {
          content.style.display = 'none';
          container.classList.remove('expanded');
          button.classList.add('collapsed');
          button.innerHTML = '<i class="fas fa-chevron-down"></i>';
        }
      }

      // Function to load OCT frame for E2E files
      async function loadE2EOCTFrame(viewportNumber, eye, frameIndex) {
        try {
          console.log(`Loading flattened OCT frame ${frameIndex} for ${eye} eye in viewport ${viewportNumber}`);
          
          // Calculate the correct frame index for flattened OCT frames
          // The backend combines frames as: dicom + oct, so we need to offset by DICOM count
          const dicomFrameCount = s3TreeData[`${eye}_eye`]?.dicom?.length || 0;
          const actualFrameIndex = dicomFrameCount + frameIndex;
          
          console.log(`Calculated frame index: ${actualFrameIndex} (DICOM: ${dicomFrameCount}, OCT: ${frameIndex})`);
          
          const response = await fetch(`/api/view_e2e_eye?dicom_file_path=${encodeURIComponent(currentE2EFile)}&eye=${eye}&frame=${actualFrameIndex}`);
          
          if (response.ok) {
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            
            const img = document.getElementById(`viewportImage${viewportNumber}`);
            const placeholder = document.querySelector(`#viewportContent${viewportNumber} .viewport-placeholder`);
            
            if (img && placeholder) {
              img.src = imageUrl;
              img.style.display = 'block';
              placeholder.style.display = 'none';
              
              // Update frame data
              const frameData = eye === 'left' ? leftEyeFrameData : rightEyeFrameData;
              frameData.currentFrame = frameIndex;
              frameData.isOctMode = true;
              
              // Update total frames for flattened OCT
              const flattenedOCTCount = s3TreeData[`${eye}_eye`]?.oct?.length || 0;
              frameData.totalFrames = flattenedOCTCount;
              
              // Store file type in viewport data for navigation
              if (!viewportData[viewportNumber]) {
                viewportData[viewportNumber] = {};
              }
              viewportData[viewportNumber].currentFileType = 'oct';
              viewportData[viewportNumber].currentEye = eye;
              viewportData[viewportNumber].dicom_file_path = currentE2EFile;
              
              // Update frame info display
              const frameInfo = document.getElementById(`frameInfo${viewportNumber}`);
              if (frameInfo) {
                frameInfo.textContent = `OCT Frame ${frameIndex + 1}`;
              }
              
              console.log(`OCT frame ${frameIndex} loaded for ${eye} eye in viewport ${viewportNumber}`);
            }
          } else {
            throw new Error(`Failed to load OCT frame: ${response.statusText}`);
          }
        } catch (error) {
          console.error(`Error loading OCT frame for ${eye} eye:`, error);
          throw error;
        }
      }

      // Function to show error in a viewport
      function showError(viewportNumber, message) {
        const errorDiv = document.getElementById(`error${viewportNumber}`);
        if (errorDiv) {
          errorDiv.textContent = message;
          errorDiv.style.display = 'block';
        }
        console.error(`Viewport ${viewportNumber} error:`, message);
      }

      // Function to update frame slider with current frame and total frames
      function updateFrameSlider(viewportNumber, currentFrame, totalFrames) {
        const frameSlider = document.getElementById(`frameSlider${viewportNumber}`);
        const frameSliderContainer = document.getElementById(`frameSliderContainer${viewportNumber}`);
        
        if (!frameSlider || !frameSliderContainer) {
          console.error(`Frame slider elements not found for viewport ${viewportNumber}`);
          return;
        }
        
        console.log(`Updating frame slider for viewport ${viewportNumber}: current=${currentFrame}, total=${totalFrames}`);
        
        // Update slider range
        frameSlider.min = 0;
        frameSlider.max = Math.max(0, totalFrames - 1);
        
        // Update slider value
        frameSlider.value = Math.min(currentFrame, totalFrames - 1);
        
        // Update frame info display
        updateFrameInfo(viewportNumber, parseInt(frameSlider.value));
        
        // Update navigation buttons
        updateNavigationButtons(viewportNumber, parseInt(frameSlider.value), totalFrames);
        
        // Show the frame slider container if it has frames
        if (totalFrames > 1) {
          frameSliderContainer.classList.add("active");
        } else {
          frameSliderContainer.classList.remove("active");
        }
        
        console.log(`Frame slider updated for viewport ${viewportNumber}`);
      }

      // Helper function to clear a viewport
      function clearViewport(viewportNumber) {
        const img = document.getElementById(`viewportImage${viewportNumber}`);
        const placeholder = document.querySelector(
          `#viewportContent${viewportNumber} .viewport-placeholder`,
        );
        const frameSliderContainer = document.getElementById(
          `frameSliderContainer${viewportNumber}`,
        );
        const errorDiv = document.getElementById(`error${viewportNumber}`);

        if (img) {
          img.style.display = "none";
          img.src = "";
        }
        if (placeholder) {
          placeholder.style.display = "block";
        }
        if (frameSliderContainer) {
          frameSliderContainer.classList.remove("active");
        }
        if (errorDiv) {
          errorDiv.style.display = "none";
        }

        // Clear viewport data
        viewportData[viewportNumber] = null;
        
        console.log(`Viewport ${viewportNumber} cleared`);
      }



      // Test function for debugging E2E loading
      function testE2ELoading() {
        console.log("Testing E2E loading...");
        console.log("selectedFilePath:", selectedFilePath);
        console.log("isE2EMode:", isE2EMode);
        console.log("currentE2EFile:", currentE2EFile);
        console.log("leftEyeFrameData:", leftEyeFrameData);
        console.log("rightEyeFrameData:", rightEyeFrameData);
        console.log("s3TreeData:", s3TreeData);
        
        // Test if we can find an E2E file in the tree
        const e2eFiles = document.querySelectorAll('.tree-item[data-file-path*=".e2e"]');
        console.log("Found E2E files in tree:", e2eFiles.length);
        if (e2eFiles.length > 0) {
          console.log("First E2E file path:", e2eFiles[0].dataset.filePath);
          selectedFilePath = e2eFiles[0].dataset.filePath;
          console.log("Set selectedFilePath to:", selectedFilePath);
        }
        
        // Test tree containers
        const leftTree = document.getElementById('leftEyeTreeContainer');
        const rightTree = document.getElementById('rightEyeTreeContainer');
        console.log("Left eye tree container:", leftTree);
        console.log("Right eye tree container:", rightTree);
        
        if (leftTree) {
          console.log("Left tree display style:", leftTree.style.display);
        }
        if (rightTree) {
          console.log("Right tree display style:", rightTree.style.display);
        }
      }

      // Test function for debugging tree rendering
      function testTreeRendering() {
        console.log("Testing tree rendering...");
        
        // Test with sample data matching the backend structure
        const sampleTreeData = {
          left_eye: {
            dicom: [
              { name: "Fundus Image 1", frame_index: 0 },
              { name: "Fundus Image 2", frame_index: 1 }
            ],
            oct: [
              { name: "Flattened OCT 1", frame_index: 0 },
              { name: "Flattened OCT 2", frame_index: 1 }
            ],
            original_oct: [
              { name: "OCT Frame 1", frame_index: 0 },
              { name: "OCT Frame 2", frame_index: 1 },
              { name: "OCT Frame 3", frame_index: 2 }
            ]
          },
          right_eye: {
            dicom: [
              { name: "Fundus Image 1", frame_index: 0 }
            ],
            oct: [
              { name: "Flattened OCT 1", frame_index: 0 }
            ],
            original_oct: [
              { name: "OCT Frame 1", frame_index: 0 },
              { name: "OCT Frame 2", frame_index: 1 }
            ]
          }
        };
        
        console.log("Rendering sample tree data:", sampleTreeData);
        renderE2ETreeData(sampleTreeData);
        
        // Show the tree containers
        showE2EControls();
      }

      // Function to debug current tree data
      function debugCurrentTreeData() {
        console.log("=== DEBUGGING CURRENT TREE DATA ===");
        console.log("s3TreeData:", s3TreeData);
        console.log("isE2EMode:", isE2EMode);
        console.log("currentE2EFile:", currentE2EFile);
        
        if (s3TreeData) {
          console.log("Tree data keys:", Object.keys(s3TreeData));
          console.log("Left eye data:", s3TreeData.left_eye);
          console.log("Right eye data:", s3TreeData.right_eye);
          
          // Check if containers exist
          const leftContainer = document.getElementById('leftEyeTreeContent');
          const rightContainer = document.getElementById('rightEyeTreeContent');
          console.log("Left container:", leftContainer);
          console.log("Right container:", rightContainer);
          
          if (leftContainer) {
            console.log("Left container innerHTML:", leftContainer.innerHTML);
          }
          if (rightContainer) {
            console.log("Right container innerHTML:", rightContainer.innerHTML);
          }
          } else {
          console.log("No tree data available");
          }
        }

      // Add click outside functionality to close the eye focus dropdown
      document.addEventListener('DOMContentLoaded', function() {
        // Close dropdown when clicking outside
        document.addEventListener('click', function(event) {
          const dropdownMenu = document.getElementById('eyeFocusDropdownMenu');
          const burgerButton = document.getElementById('eyeFocusBurgerButton');
          
          if (dropdownMenu && burgerButton) {
            const isClickInside = dropdownMenu.contains(event.target) || burgerButton.contains(event.target);
            
            if (!isClickInside && dropdownMenu.classList.contains('show')) {
              dropdownMenu.classList.remove('show');
              burgerButton.classList.remove('active');
            }
          }
          
          // Close empty file warning modal when clicking outside
          const emptyFileModal = document.getElementById('emptyFileWarningModal');
          const emptyFileForm = document.querySelector('.empty-file-warning-form');
          
          if (emptyFileModal && emptyFileForm) {
            const isClickInsideModal = emptyFileForm.contains(event.target);
            
            if (!isClickInsideModal && emptyFileModal.style.display === 'flex') {
              closeEmptyFileWarning();
            }
          }
        });
      });
