document.addEventListener('DOMContentLoaded', function() {
    const modelButtons = document.querySelectorAll('.model-btn');
    const placementCheckboxes = document.querySelectorAll('input[name="placement"]');
    const form = document.getElementById('prediction-form');
    const resultsSection = document.getElementById('results');
    const metricsContainer = document.getElementById('metrics-container');
    const comparisonViz = document.getElementById('comparison-viz');
    const placementViz = document.getElementById('placement-viz');

    let selectedModels = [];
    let selectedPlacements = [];

    // Model button selection
    modelButtons.forEach(button => {
        button.addEventListener('click', function() {
            this.classList.toggle('active');
            const model = this.dataset.model;

            if (this.classList.contains('active')) {
                if (!selectedModels.includes(model)) {
                    selectedModels.push(model);
                }
            } else {
                selectedModels = selectedModels.filter(m => m !== model);
            }
        });
    });

    // Placement type selection (enforce 2-4 selections)
    placementCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            selectedPlacements = Array.from(placementCheckboxes)
                .filter(cb => cb.checked)
                .map(cb => cb.value);

            if (selectedPlacements.length > 4) {
                this.checked = false;
                selectedPlacements = selectedPlacements.filter(p => p !== this.value);
                alert('You can select a maximum of 4 placement types.');
            }
        });
    });

    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Validate placement selection
        if (selectedPlacements.length < 2) {
            alert('Please select at least 2 placement types.');
            return;
        }

        // Gather form data
        const formData = {
            childAge: document.getElementById('child-age').value,
            childGender: document.getElementById('child-gender').value,
            childEthnicity: document.getElementById('child-ethnicity').value,
            carerAge: document.getElementById('carer-age').value,
            carerGender: document.getElementById('carer-gender').value,
            carerEthnicity: document.getElementById('carer-ethnicity').value,
            placementTypes: selectedPlacements
        };

        // Show loading state
        showLoading();

        try {
            const response = await fetch('/run_comparison', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            showError('Error running comparison: ' + error.message);
        }
    });

    function showLoading() {
        resultsSection.style.display = 'block';
        metricsContainer.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p style="margin-top: 20px; color: #4a5568;">Running model comparison...</p>
            </div>
        `;
        comparisonViz.src = '';
        placementViz.src = '';
    }

    function displayResults(data) {
        // Display metrics
        let metricsHTML = '<h3 style="margin-bottom: 20px;">Model Performance Metrics</h3>';

        if (data.metrics) {
            metricsHTML += '<div class="metric-row">';

            // Display key metrics for each model
            data.metrics.forEach(metric => {
                metricsHTML += `
                    <div class="metric-card">
                        <h4>${metric.model}</h4>
                        ${metric.accuracy !== 'N/A' ? `<div class="value">${(metric.accuracy * 100).toFixed(2)}%</div>` : ''}
                        ${metric.mse !== 'N/A' ? `<div class="value">MSE: ${metric.mse.toFixed(2)}</div>` : ''}
                    </div>
                `;
            });

            metricsHTML += '</div>';
        }

        metricsContainer.innerHTML = metricsHTML;

        // Display visualizations
        if (data.comparison_viz) {
            comparisonViz.src = data.comparison_viz + '?t=' + new Date().getTime();
        }

        if (data.placement_viz) {
            placementViz.src = data.placement_viz + '?t=' + new Date().getTime();
        }

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(message) {
        metricsContainer.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${message}
            </div>
        `;
        resultsSection.style.display = 'block';
    }
});