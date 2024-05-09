'use client'

import React, { useEffect } from 'react';
import Chart from 'chart.js/auto';


export default function AcquisitionsChart() {
  useEffect(() => {
    const data = [
      { month: 'Jan', male: 1000, female: 1200 },
      { month: 'Feb', male: 200, female: 1500 },
      { month: 'Mar', male: 1500, female: 1100 },
      { month: 'Apr', male: 250, female: 1000 },
      { month: 'May', male: 2200, female: 800 },
      { month: 'Jun', male: 300, female: 950 },
      { month: 'Jul', male: 2800 , female: 130},
    ];

    const chartConfig = {
      type: 'bar',
      data: {
        labels: data.map(row => row.month),
        datasets: [
            {
                label: 'Male',
                data: data.map(row => row.male),
                backgroundColor: 'rgba(54, 162, 235, 0.5)'
              },
            {
                label: 'Female',
                data: data.map(row => row.female),
                backgroundColor: 'rgba(255, 99, 132, 0.5)'
              },
             
        ],
      },
    };

    let chart = new Chart(document.getElementById('acquisitions').getContext('2d'), chartConfig);

    return () => {
      if (chart) 
        chart.destroy();
    };
  }, []);

  return (
    <div style={{ width: '800px' }}>
    <canvas id="acquisitions"></canvas>
  </div>
  );
};


