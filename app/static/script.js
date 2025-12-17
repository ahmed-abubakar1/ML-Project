document.getElementById('analysis-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // UI Loading State
    const btn = document.getElementById('analyze-btn');
    const btnText = btn.querySelector('.btn-text');
    const loader = document.getElementById('btn-loader');
    
    btnText.style.display = 'none';
    loader.style.display = 'block';
    btn.disabled = true;

    // Data Collection - ensure explicit conversion
    const data = {
        age: parseInt(document.getElementById('age').value) || 0,
        bmi: parseFloat(document.getElementById('bmi').value) || 0.0,
        blood_pressure: parseFloat(document.getElementById('blood_pressure').value) || 0.0,
        glucose: parseFloat(document.getElementById('glucose').value) || 0.0
    };

    // Helper for API calls
    const fetchPrediction = async (endpoint) => {
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            return await response.json();
        } catch (error) {
            console.error(error);
            return null;
        }
    };

    try {
        // Parallel Usage of all 3 endpoints
        const [heartData, costData, clusterData] = await Promise.all([
            fetchPrediction('/predict/heart-disease'),
            fetchPrediction('/predict/cost'),
            fetchPrediction('/predict/cluster')
        ]);

        // Update Heart Disease Card
        if (heartData) {
            const prob = (heartData.risk_probability * 100).toFixed(1);
            const isHighRisk = heartData.heart_disease_prediction === 1;
            
            document.getElementById('heart-prob').textContent = `${prob}%`;
            document.getElementById('heart-msg').textContent = isHighRisk 
                ? "High risk detected. Consult a cardiologist." 
                : "Profile within normal range.";
            
            const statusEl = document.getElementById('heart-status');
            const statusText = statusEl.querySelector('.status-text');
            
            statusEl.className = `status-indicator ${isHighRisk ? 'danger' : 'safe'}`;
            statusText.textContent = isHighRisk ? 'Warning' : 'Normal';
        }

        // Update Cost Card
        if (costData) {
            const cost = Math.round(costData.estimated_hospital_cost).toLocaleString();
            document.getElementById('cost-value').textContent = `$${cost}`;
        }

        // Update Cluster Card
        if (clusterData) {
            const clusterId = clusterData.patient_segment_cluster;
            document.getElementById('cluster-id').textContent = `Group ${clusterId}`;
            
            const visual = document.getElementById('cluster-visual');
            visual.className = `cluster-visual c${clusterId}`;
            
            const descs = [
                "Low Risk / Standard Profile",
                "Medium Risk / Monitor Closely",
                "High Risk / Immediate Attention"
            ];
            // Safe fallback if cluster > 2
            document.getElementById('cluster-desc').textContent = descs[clusterId] || "Unknown Segment";
        }

    } catch (err) {
        alert("An error occurred during analysis.");
    } finally {
        // Reset UI
        btnText.style.display = 'block';
        loader.style.display = 'none';
        btn.disabled = false;
    }
});
