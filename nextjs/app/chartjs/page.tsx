'use client'

import React, { useEffect, useState, useRef } from 'react';
import Chart, {ChartItem} from 'chart.js/auto';


function init_score_chart( criteria : string )  {
    let data: any[] = []
    let config = {type: 'bar', data: {}}
    if (criteria === '성별') {
        data = [
            {score: '1점', male: 1000, female: 1200},
            {score: '2점', male: 200, female: 1500},
            {score: '3점', male: 1500, female: 1100},
            {score: '4점', male: 250, female: 1000},
            {score: '5점', male: 2200, female: 800},
        ];
        config['data'] = {
            labels: data.map(row => row.score),
            datasets: [
                {
                    label: '남성',
                    data: data.map(row => row.male),
                    backgroundColor: 'rgba(54, 162, 235, 0.5)'
                },
                {
                    label: '여성',
                    data: data.map(row => row.female),
                    backgroundColor: 'rgba(255, 99, 132, 0.5)'
                }
            ],
        }
    } else if (criteria === '연령') {
        data = [
            {score: '1점', twenty: 1000, thirty: 400, forty: 400, fifty: 800, sixty_over: 400},
            {score: '2점', twenty: 400, thirty: 500, forty: 200, fifty: 400, sixty_over: 200},
            {score: '3점', twenty: 500, thirty: 600, forty: 500, fifty: 200, sixty_over: 600},
            {score: '4점', twenty: 800, thirty: 700, forty: 1200, fifty: 1000, sixty_over: 700},
            {score: '5점', twenty: 900, thirty: 800, forty: 1300, fifty: 900, sixty_over: 700}
        ];
        config['data'] = {
            labels: data.map(row => row.score),
            datasets: [
                {
                    label: '20대',
                    data: data.map(row => row.twenty),
                    backgroundColor: 'rgba(54, 162, 235, 0.5)'
                },
                {
                    label: '30대',
                    data: data.map(row => row.thirty),
                    backgroundColor: 'rgba(255, 99, 132, 0.5)'
                },
                {
                    label: '40대',
                    data: data.map(row => row.forty),
                    backgroundColor: 'rgba(255, 205, 86, 0.5)'
                },
                {
                    label: '50대',
                    data: data.map(row => row.fifty),
                    backgroundColor: 'rgba(75, 192, 192, 0.5)'
                },
                {
                    label: '60대 이상',
                    data: data.map(row => row.sixty_over),
                    backgroundColor: 'rgba(153, 102, 255, 0.5)'
                }
            ],
        }
    }
    const chartStatus = Chart.getChart('score_chart');
    if (chartStatus !== undefined) 
        chartStatus.destroy();
    const canvasElement = document.getElementById('score_chart') as HTMLCanvasElement;
    // @ts-ignore
    let chart = new Chart(canvasElement.getContext('2d'), config);

    return () => {
        if (chart)
            chart.destroy();
    };
}

function ScoreChartPanel() {
    const [criteria, setCriteria] = useState('성별');
    
      useEffect(() => {
          init_score_chart(criteria);
      }, [criteria]);
    
    const handleOptionChange = (event: { target: { value: React.SetStateAction<string>; }; }) => {
      setCriteria(event.target.value);
    };
    
    return (
      <div style={{ width: '800px', backgroundColor: 'white', borderRadius: '10px', padding: '20px', boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)' }}>
        <div style={{ marginBottom: '20px' }}>
          <select id="criteria" value={criteria} onChange={handleOptionChange}>
            <option value="성별">성별</option>
            <option value="연령">연령</option>
          </select>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <canvas id="score_chart" style={{ border: '1px solid black', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)' }}></canvas>
        </div>
      </div>
    );
}
  
export default ScoreChartPanel;


