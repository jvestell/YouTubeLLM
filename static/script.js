document.addEventListener('DOMContentLoaded', (event) => {
    const fetchButton = document.getElementById('fetch-videos');
    const summarizeButton = document.getElementById('summarize');
    const searchButton = document.getElementById('search');
    const generateQuestionsButton = document.getElementById('generate-questions');
    const form = document.getElementById('video-search-form');
    const resultsSection = document.getElementById('results');
    const summarySection = document.getElementById('summary');
    const searchResultsSection = document.getElementById('search-results');
    const questionsSection = document.getElementById('questions');

    fetchButton.addEventListener('click', fetchVideos);
    summarizeButton.addEventListener('click', summarizeVideos);
    searchButton.addEventListener('click', searchVideos);
    generateQuestionsButton.addEventListener('click', generateQuestions);

    function handleResponse(response) {
        if (!response.ok) {
            return response.json().then(err => { throw err; });
        }
        return response.json();
    }

    function showProcessing(show) {
        let overlay = document.getElementById('processing-overlay');
        if (show) {
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'processing-overlay';
                overlay.innerHTML = '<div class="processing-content"><p>Processing...</p></div>';
                document.body.appendChild(overlay);
            }
            overlay.style.display = 'flex';
        } else {
            if (overlay) {
                overlay.style.display = 'none';
            }
        }
    }

    function fetchVideos(event) {
        event.preventDefault(); // Prevent the default form submission
        showProcessing(true);
        const formData = new FormData(form);
        fetch('/fetch_videos', {
            method: 'POST',
            body: formData
        })
        .then(handleResponse)
        .then(data => {
            showProcessing(false);
            displayVideos(data.videos);
        })
        .catch(error => {
            showProcessing(false);
            console.error('Error:', error);
            displayError(error.error || 'An unexpected error occurred while fetching videos.');
        });
    }

    function summarizeVideos() {
        showProcessing(true);
        fetch('/summarize_videos')
        .then(handleResponse)
        .then(data => {
            showProcessing(false);
            displaySummaries(data.summaries);
        })
        .catch(error => {
            showProcessing(false);
            console.error('Error:', error);
            displayError(error.error || 'An unexpected error occurred while summarizing videos.');
        });
    }

    function searchVideos() {
        const query = prompt("Enter your search query:");
        if (query) {
            showProcessing(true);
            fetch('/search_videos', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({query: query})
            })
            .then(handleResponse)
            .then(data => {
                showProcessing(false);
                displaySearchResults(data.results);
            })
            .catch(error => {
                showProcessing(false);
                console.error('Error:', error);
                displayError(error.error || 'An unexpected error occurred while searching videos.');
            });
        }
    }

    function generateQuestions() {
        showProcessing(true);
        fetch('/generate_questions')
        .then(handleResponse)
        .then(data => {
            showProcessing(false);
            displayQuestions(data.questions);
        })
        .catch(error => {
            showProcessing(false);
            console.error('Error:', error);
            displayError(error.error || 'An unexpected error occurred while generating questions.');
        });
    }

    function displayVideos(videos) {
        resultsSection.innerHTML = '<h2>Fetched Videos</h2>';
        videos.forEach(video => {
            resultsSection.innerHTML += `
                <div class="video-item">
                    <h3>${video.Title}</h3>
                    <p>Views: ${video.ViewCount}</p>
                    <p><a href="${video.Link}" target="_blank">Watch Video</a></p>
                </div>
            `;
        });
        resultsSection.style.display = 'block';
    }

    function displaySummaries(summaries) {
        summarySection.innerHTML = '<h2>Video Summaries</h2>';
        summaries.forEach((summary, index) => {
            summarySection.innerHTML += `
                <div class="summary-item">
                    <h3>Summary ${index + 1}</h3>
                    <p>${summary}</p>
                </div>
            `;
        });
        summarySection.style.display = 'block';
    }

    function displaySearchResults(results) {
        searchResultsSection.innerHTML = '<h2>Search Results</h2>';
        results.forEach(result => {
            searchResultsSection.innerHTML += `
                <div class="search-result-item">
                    <h3>${result.title}</h3>
                    <p>Laughter Count: ${result.laughter_count}</p>
                    <p>Laughter Intensity: ${result.laughter_intensity}</p>
                    <p>Applause Count: ${result.applause_count}</p>
                    <p>Similarity Score: ${result.similarity_score}</p>
                    <h4>Funny Moments:</h4>
                    <ul>
                        ${result.funny_moments.map(moment => `
                            <li>${moment.timestamp}: ${moment.text}</li>
                        `).join('')}
                    </ul>
                    <p><a href="${result.link}" target="_blank">Watch Video</a></p>
                </div>
            `;
        });
        searchResultsSection.style.display = 'block';
    }

    function displayQuestions(questions) {
        questionsSection.innerHTML = '<h2>Generated Questions</h2><ul>';
        questions.forEach(question => {
            questionsSection.innerHTML += `<li>${question}</li>`;
        });
        questionsSection.innerHTML += '</ul>';
        questionsSection.style.display = 'block';
    }

    function displayError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        document.body.insertBefore(errorDiv, document.body.firstChild);
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
});