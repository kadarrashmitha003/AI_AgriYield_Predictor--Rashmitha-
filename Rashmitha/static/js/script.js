// AgriYield Predictor JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Sample data presets
    const sampleData = {
        'rice': {
            nitrogen: 90,
            phosphorus: 42,
            potassium: 43,
            temperature: 25.0,
            humidity: 82.0,
            ph: 6.5,
            rainfall: 220.0
        },
        'maize': {
            nitrogen: 70,
            phosphorus: 50,
            potassium: 40,
            temperature: 22.0,
            humidity: 65.0,
            ph: 6.0,
            rainfall: 80.0
        },
        'chickpea': {
            nitrogen: 40,
            phosphorus: 70,
            potassium: 80,
            temperature: 18.0,
            humidity: 60.0,
            ph: 7.0,
            rainfall: 65.0
        }
    };

    // Load sample data when sample cards are clicked
    const sampleCards = document.querySelectorAll('.sample-card');
    sampleCards.forEach(card => {
        card.addEventListener('click', function() {
            const sampleType = this.getAttribute('data-sample');
            const data = sampleData[sampleType];
            
            if (data) {
                document.getElementById('nitrogen').value = data.nitrogen;
                document.getElementById('phosphorus').value = data.phosphorus;
                document.getElementById('potassium').value = data.potassium;
                document.getElementById('temperature').value = data.temperature;
                document.getElementById('humidity').value = data.humidity;
                document.getElementById('ph').value = data.ph;
                document.getElementById('rainfall').value = data.rainfall;
                
                // Show success message
                // Scroll to form
                document.getElementById('predictionForm').scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // Form validation and submission
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            const submitBtn = document.getElementById('predictBtn');
            const loadingSpinner = document.getElementById('loadingSpinner');
            
            // Show loading spinner
            // Disable submit button
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
            
            // Show loading message
            loadingSpinner.style.display = 'block';
            
            // Validate form inputs
            if (!validateForm()) {
                e.preventDefault();
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="fas fa-calculator"></i> Predict Best Crop';
                loadingSpinner.style.display = 'none';
                return false;
            }
        });
    }

    // Form validation function
    function validateForm() {
        const inputs = [
            'nitrogen', 'phosphorus', 'potassium', 
            'temperature', 'humidity', 'ph', 'rainfall'
        ];
        
        let isValid = true;
        
        inputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            const value = parseFloat(input.value);
            const min = parseFloat(input.min);
            const max = parseFloat(input.max);
            
            if (isNaN(value) || value < min || value > max) {
                input.classList.add('is-invalid');
                isValid = false;
            } else {
                input.classList.remove('is-invalid');
            }
        });
        
        return isValid;
    }

    // Real-time input validation
    const formInputs = document.querySelectorAll('#predictionForm input[type="number"]');
    formInputs.forEach(input => {
        input.addEventListener('input', function() {
            const value = parseFloat(this.value);
            const min = parseFloat(this.min);
            const max = parseFloat(this.max);
            
            if (isNaN(value) || value < min || value > max) {
                this.classList.add('is-invalid');
            } else {
                this.classList.remove('is-invalid');
            }
        });
    });

    // Add tooltips to form inputs
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Add animation to elements when they come into view
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate__animated', 'animate__fadeInUp');
            }
        });
    }, observerOptions);

    // Observe elements for animation
    document.querySelectorAll('.feature-card, .step-card, .tech-card').forEach(el => {
        observer.observe(el);
    });

    // API test function (for development)
    window.testAPI = function() {
        const testData = {
            N: 90,
            P: 40,
            K: 40,
            temperature: 25,
            humidity: 80,
            ph: 6.5,
            rainfall: 200
        };

        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(testData)
        })
        .then(response => response.json())
        .then(data => {
            console.log('API Response:', data);
            if (data.success) {
                alert(`API Test Successful! Predicted: ${data.predicted_crop} (${data.confidence}% confidence)`);
            } else {
                alert('API Test Failed: ' + data.error);
            }
        })
        .catch(error => {
            console.error('API Test Error:', error);
            alert('API Test Failed: ' + error.message);
        });
    };
});

// Utility function to format numbers
function formatNumber(num, decimals = 1) {
    return parseFloat(num).toFixed(decimals);
}

// Utility function to show notifications
function showNotification(message, type = 'success') {
    const alertClass = type === 'success' ? 'alert-success' : 'alert-danger';
    const notification = document.createElement('div');
    notification.className = `alert ${alertClass} alert-dismissible fade show`;
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.prepend(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}