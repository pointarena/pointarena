document.addEventListener('DOMContentLoaded', function() {
    // Navbar active link highlighting
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Model filtering functionality
    const modelTypeFilter = document.querySelector('.filter-group select:first-of-type');
    const providerFilter = document.querySelector('.filter-group select:last-of-type');
    const modelCards = document.querySelectorAll('.model-card');

    if (modelTypeFilter && providerFilter) {
        const filterModels = () => {
            const typeValue = modelTypeFilter.value;
            const providerValue = providerFilter.value;
            
            modelCards.forEach(card => {
                const modelType = card.querySelector('.badge:first-of-type').textContent;
                const provider = card.querySelector('.badge:last-of-type').textContent;
                
                const typeMatch = typeValue === 'All Types' || modelType === typeValue;
                const providerMatch = providerValue === 'All Providers' || provider.includes(providerValue);
                
                card.closest('.col-md-4').style.display = typeMatch && providerMatch ? 'block' : 'none';
            });
        };
        
        modelTypeFilter.addEventListener('change', filterModels);
        providerFilter.addEventListener('change', filterModels);
    }

    // Leaderboard sorting functionality for all ranking tables
    document.querySelectorAll('.leaderboard-table').forEach(leaderboardTable => {
        const headers = leaderboardTable.querySelectorAll('thead th');
        let currentSortCol = 0; // Default sort by rank
        let sortDirection = 'asc';
        
        headers.forEach((header, index) => {
            // Skip first column (rank) as it's already sorted
            if (index === 0) return;
            
            header.style.cursor = 'pointer';
            header.addEventListener('click', () => {
                sortTable(leaderboardTable, index);
            });
            
            // Add sort indicator
            const sortIndicator = document.createElement('span');
            sortIndicator.classList.add('sort-indicator');
            sortIndicator.style.marginLeft = '5px';
            sortIndicator.innerHTML = '↕';
            header.appendChild(sortIndicator);
        });
        
        function sortTable(table, columnIndex) {
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Reset all indicators
            headers.forEach(h => {
                const indicator = h.querySelector('.sort-indicator');
                if (indicator) indicator.innerHTML = '↕';
            });
            
            // Set new indicator
            const indicator = headers[columnIndex].querySelector('.sort-indicator');
            
            // Toggle sort direction if clicking on same column
            if (currentSortCol === columnIndex) {
                sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                sortDirection = 'asc';
                currentSortCol = columnIndex;
            }
            
            // Update indicator
            indicator.innerHTML = sortDirection === 'asc' ? '↑' : '↓';
            
            // Sort the rows
            rows.sort((a, b) => {
                let aValue = a.cells[columnIndex].textContent;
                let bValue = b.cells[columnIndex].textContent;
                
                // Handle numeric values
                if (!isNaN(parseFloat(aValue))) {
                    aValue = parseFloat(aValue);
                    bValue = parseFloat(bValue);
                }
                
                if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
                if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
                return 0;
            });
            
            // Re-append rows in new order
            rows.forEach(row => tbody.appendChild(row));
        }
    });

    // Card hover effects enhancement
    const cards = document.querySelectorAll('.model-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px)';
            this.style.boxShadow = '0 12px 20px rgba(0, 0, 0, 0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
        });
    });

    // Add a small animation to hero section
    const heroContent = document.querySelector('.hero-section .container');
    if (heroContent) {
        heroContent.style.opacity = '0';
        heroContent.style.transform = 'translateY(20px)';
        heroContent.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
        
        setTimeout(() => {
            heroContent.style.opacity = '1';
            heroContent.style.transform = 'translateY(0)';
        }, 300);
    }

    // Add scroll reveal animation to sections
    const sections = document.querySelectorAll('section:not(.hero-section)');
    const revealSection = (entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                observer.unobserve(entry.target);
            }
        });
    };
    
    const sectionObserver = new IntersectionObserver(revealSection, {
        root: null,
        threshold: 0.15,
    });
    
    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
        
        sectionObserver.observe(section);
    });
    
    document.addEventListener('scroll', () => {
        sections.forEach(section => {
            const sectionTop = section.getBoundingClientRect().top;
            const windowHeight = window.innerHeight;
            
            if (sectionTop < windowHeight * 0.85) {
                section.style.opacity = '1';
                section.style.transform = 'translateY(0)';
            }
        });
    });

    // CSV Data Loading functionality
    const csvDownloadButtons = document.querySelectorAll('.csv-download .btn');
    csvDownloadButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const section = this.closest('section');
            const sectionId = section.id;
            
            // Define CSV file paths based on section ID
            let csvPath;
            switch(sectionId) {
                case 'point-bench':
                    csvPath = 'data/point-bench.csv';
                    break;
                case 'point-battle':
                    csvPath = 'data/point-battle.csv';
                    break;
                case 'point-act':
                    csvPath = 'data/point-act.csv';
                    break;
                default:
                    console.error('Unknown section ID');
                    return;
            }
            
            // Download the CSV file
            downloadCSV(csvPath);
        });
    });
    
    function downloadCSV(filePath) {
        fetch(filePath)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.blob();
            })
            .then(blob => {
                // Create a temporary URL for the blob
                const url = window.URL.createObjectURL(blob);
                
                // Create a temporary anchor element and trigger download
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = filePath.split('/').pop();
                document.body.appendChild(a);
                a.click();
                
                // Clean up
                window.URL.revokeObjectURL(url);
                a.remove();
            })
            .catch(error => {
                console.error('Error downloading CSV:', error);
                alert('Failed to download CSV file. Please try again later.');
            });
    }

    // Function to create directory structure for CSV files if implementing
    function createDataDirectories() {
        // This would typically be done on the server side
        // For GitHub Pages, you would manually create these directories and files
        console.log('Data directories should be created manually for GitHub Pages');
    }

    // Function to parse CSV data
    function parseCSV(csv) {
        const lines = csv.trim().split('\n');
        const headers = lines[0].split(',');
        
        return lines.slice(1).map(line => {
            const values = line.split(',');
            const entry = {};
            
            headers.forEach((header, index) => {
                entry[header.trim()] = values[index]?.trim() || '';
            });
            
            return entry;
        });
    }

    // Function to load CSV data
    async function loadCSV(url) {
        const response = await fetch(url);
        const data = await response.text();
        return parseCSV(data);
    }

    // Function to safely parse numeric values
    function safeParseFloat(value, fallback = 0) {
        if (value === undefined || value === null || value === '') {
            return fallback;
        }
        
        // Remove any % character if present
        if (typeof value === 'string' && value.includes('%')) {
            value = value.replace('%', '');
        }
        
        const parsed = parseFloat(value);
        return isNaN(parsed) ? fallback : parsed;
    }

    // Function to update Point-Bench leaderboard
    async function updatePointBenchTable() {
        try {
            const data = await loadCSV('data/point-bench.csv');
            const tableBody = document.querySelector('#point-bench .leaderboard-table tbody');
            tableBody.innerHTML = '';
            
            // Sort by Average score (descending)
            data.sort((a, b) => safeParseFloat(b.Average) - safeParseFloat(a.Average));
            
            // Only display top 5 results
            data.forEach((row, index) => {
                const tr = document.createElement('tr');
                
                tr.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${row[''] || row.Model || ''}</td>
                    <td class="score">${safeParseFloat(row.Affordance).toFixed(2)}</td>
                    <td class="score">${safeParseFloat(row.Spatial).toFixed(2)}</td>
                    <td class="score">${safeParseFloat(row.Reasoning).toFixed(2)}</td>
                    <td class="score">${safeParseFloat(row.Steerability).toFixed(2)}</td>
                    <td class="score">${safeParseFloat(row.Counting).toFixed(2)}</td>
                    <td class="score">${safeParseFloat(row.Average).toFixed(3)}</td>
                `;
                
                tableBody.appendChild(tr);
            });
            
            // Hide loading indicator
            document.querySelector('#point-bench .loading-indicator').style.display = 'none';
        } catch (error) {
            console.error('Error loading point-bench.csv:', error);
            document.querySelector('#point-bench .loading-indicator').innerHTML = 
                '<p class="text-danger">Error loading data. Please try again later.</p>';
        }
    }

    // Function to update Point-Battle leaderboard
    async function updatePointBattleTable() {
        try {
            const data = await loadCSV('data/point-battle.csv');
            const tableBody = document.querySelector('#point-battle .leaderboard-table tbody');
            tableBody.innerHTML = '';
            
            // Sort by Elo Rating (descending) just to be sure
            data.sort((a, b) => safeParseFloat(b['Elo Rating']) - safeParseFloat(a['Elo Rating']));
            
            // Display all available results
            data.forEach((row, index) => {
                const tr = document.createElement('tr');
                
                tr.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${row.Model || ''}</td>
                    <td class="score">${safeParseFloat(row['Elo Rating']).toFixed(1)}</td>
                    <td class="score">${safeParseFloat(row.Wins, 0)}</td>
                    <td class="score">${safeParseFloat(row.Losses, 0)}</td>
                    <td class="score">${row['Win Rate'] || '0%'}</td>
                `;
                
                tableBody.appendChild(tr);
            });
            
            // Hide loading indicator
            document.querySelector('#point-battle .loading-indicator').style.display = 'none';
        } catch (error) {
            console.error('Error loading point-battle.csv:', error);
            document.querySelector('#point-battle .loading-indicator').innerHTML = 
                '<p class="text-danger">Error loading data. Please try again later.</p>';
        }
    }

    // Function to update Point-Act leaderboard
    async function updatePointActTable() {
        try {
            const data = await loadCSV('data/point-act.csv');
            const tableBody = document.querySelector('#point-act .leaderboard-table tbody');
            tableBody.innerHTML = '';
            
            // Sort by Success Rate (descending)
            data.sort((a, b) => safeParseFloat(b['Success Rate']) - safeParseFloat(a['Success Rate']));
            
            data.forEach((row) => {
                const tr = document.createElement('tr');
                
                tr.innerHTML = `
                    <td>${row.Challenge || ''}</td>
                    <td class="score">${safeParseFloat(row['Success Rate'])}</td>
                    <td class="score">${safeParseFloat(row['SUS Score'])}</td>
                `;
                
                tableBody.appendChild(tr);
            });
            
            // Hide loading indicator
            document.querySelector('#point-act .loading-indicator').style.display = 'none';
        } catch (error) {
            console.error('Error loading point-act.csv:', error);
            document.querySelector('#point-act .loading-indicator').innerHTML = 
                '<p class="text-danger">Error loading data. Please try again later.</p>';
        }
    }

    // Initialize all tables
    updatePointBenchTable();
    updatePointBattleTable();
    updatePointActTable();
}); 