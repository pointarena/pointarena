# Point Arena

This is a GitHub Pages site for Point Arena, a comprehensive AI model benchmarking platform that evaluates models across multiple dimensions.

## Features

- Three distinct benchmark categories: Point-Bench, Point-Battle, and Point-Act
- Responsive design based on Bootstrap 5
- Interactive model cards with hover effects
- Sortable leaderboard tables
- Model filtering functionality
- CSV data download capabilities
- Smooth scrolling navigation
- Mobile-friendly layout

## Benchmarks

### Point-Bench
Comprehensive benchmark evaluating model general capabilities across various tasks including reasoning, knowledge, coding, and safety.

### Point-Battle
Head-to-head model comparison in adversarial scenarios, measuring prompt robustness, jailbreak resistance, and handling of adversarial inputs.

### Point-Act
Real-world task execution and agent capabilities evaluation, focusing on task completion, tool usage, planning, and decision making.

## Setup Instructions

### Setting up GitHub Pages

1. Fork this repository to your GitHub account
2. Go to Settings > Pages
3. Under "Source", select "main" branch
4. Click "Save"
5. Your site will be published at `https://[yourusername].github.io/pointarena`

### Running Locally

1. Clone this repository to your local machine
   ```
   git clone https://github.com/pointarena/pointarena.git
   ```
2. Open the `index.html` file in your browser

## Data Files

The benchmark data is stored in three CSV files:
- `data/point-bench.csv` - General benchmarking data
- `data/point-battle.csv` - Adversarial comparison data
- `data/point-act.csv` - Task execution data

To update the data, simply replace these CSV files with your updated data, keeping the same column structure.

## Customization

### Modifying Content

- Edit the `index.html` file to change the text content, add or remove models, or modify the leaderboard data
- The site uses Bootstrap 5 classes for styling and layout

### Styling Changes

- The main styling is in `styles.css`
- Color scheme is defined using CSS variables at the top of the file

### Adding Functionality

- The interactive elements are controlled by JavaScript in `script.js`
- CSV data loading and table sorting functionality are included

## Structure

- `index.html` - Main HTML file with the site structure
- `styles.css` - CSS styles for the site
- `script.js` - JavaScript for interactive elements and data loading
- `data/` - Directory containing CSV files for each benchmark
- `README.md` - This documentation file

## License

MIT License - Feel free to use this template for your own projects.
