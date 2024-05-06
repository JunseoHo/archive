'use client'

import { ResponsiveBar } from '@nivo/bar'

export default function nivo_page() {

  var data = [
    {
      "month": "Jan",
      "male": 1100,
      "maleColor": "hsl(206, 100%, 87%)",
      "female": 800,
      "femaleColor": "hsl(0, 100%, 87%)",
    },
    {
      "month": "Feb",
      "male": 1200,
      "maleColor": "hsl(206, 100%, 87%)",
      "female": 1300,
      "femaleColor": "hsl(0, 100%, 87%)",
    },
    {
      "month": "Mar",
      "male": 900,
      "maleColor": "hsl(206, 100%, 87%)",
      "female": 910,
      "femaleColor": "hsl(0, 100%, 87%)",
    },
    {
      "month": "Apr",
      "male": 500,
      "maleColor": "hsl(206, 100%, 87%)",
      "female": 1400,
      "femaleColor": "hsl(0, 100%, 87%)",
    },
    {
      "month": "May",
      "male": 600,
      "maleColor": "hsl(206, 100%, 87%)",
      "female": 800,
      "femaleColor": "hsl(0, 100%, 87%)",
    }
  ]

  return(
    <ResponsiveBar
        data={data}
        keys={[
            'male',
            'female'
        ]}
        indexBy="month"
        margin={{ top: 50, right: 130, bottom: 50, left: 60 }}
        padding={0.3}
        valueScale={{ type: 'linear' }}
        indexScale={{ type: 'band', round: true }}
        colors={{ scheme: 'nivo' }}
        defs={[
            {
                id: 'dots',
                type: 'patternDots',
                background: 'inherit',
                color: '#d5b191',
                size: 4,
                padding: 1,
                stagger: true
            },
            {
                id: 'lines',
                type: 'patternLines',
                background: 'inherit',
                color: '#F5A58C',
                rotation: -45,
                lineWidth: 5,
                spacing: 10
            }
        ]}
        fill={[
            {
                match: {
                    id: 'male'
                },
                id: 'dots'
            },
            {
                match: {
                    id: 'female'
                },
                id: 'lines'
            }
        ]}
        borderColor={{
            from: 'color',
            modifiers: [
                [
                    'darker',
                    1.6
                ]
            ]
        }}
        axisTop={null}
        axisRight={null}
        axisBottom={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: 'Month',
            legendPosition: 'middle',
            legendOffset: 32,
            truncateTickAt: 0
        }}
        axisLeft={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: 'Reviews',
            legendPosition: 'middle',
            legendOffset: -40,
            truncateTickAt: 0
        }}
        labelSkipWidth={12}
        labelSkipHeight={12}
        labelTextColor={{
            from: 'color',
            modifiers: [
                [
                    'darker',
                    1.6
                ]
            ]
        }}
        legends={[
            {
                dataFrom: 'keys',
                anchor: 'bottom-right',
                direction: 'column',
                justify: false,
                translateX: 120,
                translateY: 0,
                itemsSpacing: 2,
                itemWidth: 100,
                itemHeight: 20,
                itemDirection: 'left-to-right',
                itemOpacity: 0.85,
                symbolSize: 20,
                effects: [
                    {
                        on: 'hover',
                        style: {
                            itemOpacity: 1
                        }
                    }
                ]
            }
        ]}
        role="application"
        ariaLabel="reviewer_demographics"
    />
  )
}
