---
title: "About"
layout: single
permalink: /about/
author_profile: false
header:
    overlay_image: /assets/logos/background.png
---
{% include author-profile-about.html %}

<div class="intro">
<p align="center">
"데이터를 기반으로 보다 <span class="word-highlight">안전한 사회</span>, 나아가 더 나은 세상에 기여하고 싶다는 가치관을 가지고 있습니다. 꾸준히 데이터사이언스와 통계학을 공부하고 있습니다. 이 사이트는 평소 공부하는 자료를 정리해서 업로드하는 블로그로 사용하고 있습니다. 수정이 필요한 사항 또는 어떤 사소한 내용이든 댓글 모두 환영합니다😀"
</p>
</div>

# 📚 Interest
## Data-based Public Safety

- Smart city (IoT data)
- Spatial Data Analysis
- Crime Pattern Analysis
- Transportation Data Analysis

# 🎓 Education 

- Department of Statistics, Seoul National University - 2022.03 ~
    - M.S in Statistics
    - Spatial Statistics Lab. (Advisor: Prof. Lim Chae Young)
- Korea National Police University - 2021.02
    - Bachelor of Public Administration
    - Bachelor of Police Science

# 🥇 Projects and Awards
- **2nd Prize**, Safety Data Analysis Competition - 2023.02
  - Developed a prediction model for voice phishing crime risk areas
  - Used space-time kernel density estimation and spatial lag regression model      
    
- **3rd Prize**, Seoul IoT city data hackathon - 2022.07
  - Analyzed accident risk factors and predicted accident severity for personal mobility devices (PMs)
  - Used S-DoT city data and ordinal regression model as methodology

# 👮‍♂️ Experience
- Korea National Police Agency - 2021.03 ~
  - Police Officer
  - Platoon Leader of Auxiliary Police, Seoul Metropolitan Police Agency
- Teaching Assistant, Seoul National University
  - Statistics Lab - Spring 2023
  - Introduction to Data Science - Fall 2023


<style>
  body {
    word-break: keep-all;
  }
  h1 {
    /* font-size: 1.5em !important; */
    /* border-bottom: none !important; */
    margin-bottom: 0.5em !important;

  }
  h2 {
    font-size: 1.25em;
    font-weight: normal !important;
    border-bottom: none !important;
    margin-top: 0em !important;
    margin-bottom: 0em !important;
  }
  h3 {
    font-size: 1em;
  }
  ul {
    font-size: 0.9em !important;
    margin-top: 0em !important;
  }

  ul ul {
    margin-top: 0.5em !important;
    font-size: 0.8em !important; 
    text-indent: 1em !important;
    padding-left: 1em !important;
  }

  .word-highlight {
    font-weight: bold;
    font-size: 1em !important;
    display: inline-block;
    position: relative;
    width: fit-content;
  }

  .word-highlight::after{
    content: "";
    width: 0; /* Initially, the highlight starts with no width */
    height: 100%;
    background-color: rgba(155,251,225,0.5);    
    position: absolute;
    left: 0;
    transition: width 0.3s ease-in-out; /* Transition width property */
    pointer-events: none;
    z-index: -1;
}

.intro {
  cursor: pointer;
}

.intro:hover .word-highlight::after {
  width: 100%;
}
</style>