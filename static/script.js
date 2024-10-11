document.addEventListener('DOMContentLoaded', (event) => {
    const fetchButton = document.getElementById('fetch-videos');
    const showPreferredMomentsButton = document.getElementById('show-preferred-moments');
    const form = document.getElementById('video-search-form');
    const resultsSection = document.getElementById('results');
    const preferredMomentsSection = document.getElementById('preferred-moments');

    fetchButton.addEventListener('click', fetchVideos);
    showPreferredMomentsButton.addEventListener('click', showPreferredMoments);

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

    function showPreferredMoments() {
        showProcessing(true);
        fetch('/show_preferred_moments')
        .then(handleResponse)
        .then(data => {
            showProcessing(false);
            displayPreferredMoments(data.preferred_moments);
        })
        .catch(error => {
            showProcessing(false);
            console.error('Error:', error);
            displayError(error.error || 'An unexpected error occurred while showing preferred moments.');
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
        preferredMomentsSection.style.display = 'none';
    }

    function displayPreferredMoments(moments) {
        preferredMomentsSection.innerHTML = '<h2>Preferred Moments</h2>';
        moments.forEach((moment, index) => {
            const videoUrl = `https://www.youtube.com/watch?v=${moment.video_id}&t=${moment.timestamp}`;
            preferredMomentsSection.innerHTML += `
                <div class="preferred-moment-item">
                    <h3>Moment ${index + 1}</h3>
                    <p>${moment.summary}</p>
                    <p><a href="${videoUrl}" target="_blank">Watch this moment</a></p>
                </div>
            `;
        });
        preferredMomentsSection.style.display = 'block';
        resultsSection.style.display = 'none';
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