# 📊 AIRBNBS_LISTING_NAIROBI

## 📁 Project Overview
This project analyzes Airbnb listings in Nairobi to understand how listing characteristics, such as room type, cancellation policy, minimum stay requirements, and professional management, influence pricing and host behavior. The analysis matters because short-term rentals play a growing role in Nairobi’s housing and tourism ecosystem, affecting affordability, availability, and guest experience. By examining these patterns, the project provides data-driven insights into market dynamics and host strategies within Nairobi’s short-term rental market.

---

## 🎯 Objectives
- Identify trends and patterns  
- Perform exploratory data analysis (EDA)  
- Build predictive models (if applicable)  
- Generate business or policy insights  

---

## 📂 Dataset

### Description

This dataset contains comprehensive short-term rental analytics for the Airbnb market in Nairobi, Kenya. It provides insights into property performance, listing characteristics, host management strategies, and revenue metrics for the local vacation rental market.

### Dataset Overview

*   **Number of Rows:** 300 (active listings)
    
*   **Number of Columns:** 66+ (comprehensive listing attributes)
    
*   **Time Period Covered:** Yearly snapshot (12-month trailing data) and last 3 months performance data
    
*   **Last Updated:** December 15, 2025
    

### Key Features

*   **Property Attributes:** listing\_type, room\_type, bedrooms, amenities, listing\_name
    
*   **Host Details:** host\_id, professional\_management
    
*   **Performance Metrics:** revenue\_per\_year, avg\_rate\_per\_night, minimum\_nights, occupancy\_rate
    
*   **Booking Policies:** cancellation\_policy
    
*   **Visual Content:** cover\_photo\_url
    
*   **Ratings:** Review scores (if available)
    

### Data Structure

The dataset is organized with the following key segments:

*   **Listing Information:** Property type, room configuration, and amenities
    
*   **Host Management:** Professional vs. individual host classification
    
*   **Financial Metrics:** Annual revenue, nightly rates, and revenue per night
    
*   **Booking Constraints:** Minimum night requirements and cancellation policies
    
*   **Market Positioning:** Listing types and room categories
    

### Feature Examples

*   listing\_type (Categorical): Property type (e.g., "Entire rental Unit", "Private room", "Treehouse")
    
*   room\_type (Categorical): Room category ("Entire home/apartment", "Private room", "Hotel room")
    
*   avg\_rate\_per\_night (Numerical): Average nightly rate charged
    
*   revenue\_per\_year (Numerical): Total annual revenue generated
    
*   cancellation\_policy (Categorical): Host's cancellation terms (Flexible, Moderate, Firm, Strict, Limited)
    
*   professional\_management (Boolean): Whether the listing is professionally managed
    
*   minimum\_nights (Numerical): Minimum night stay requirement

### Dataset Source
https://www.airroi.com/data-portal/markets/nairobi-kenya
---

## 🛠️ Tools & Technologies
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)  
- Jupyter Notebook  
- SQL  
- Power BI

---

## 📈 Analysis Workflow
1. Data loading  
2. Cleaning & preprocessing  
3. Exploratory data analysis (EDA)  
4. Feature engineering  
5. Modeling (optional)  
6. Insights & interpretation  
7. Visualizations  

---

## 📊 Key Insights
### Pricing & Revenue Analysis

*   What's the correlation between listing type and average nightly rate? tick
    
*   Which room types generate the most revenue despite having lower average rates? tick
    
*   Are professionally managed listings priced higher than individually managed ones? tick
    
*   What price range attracts the most bookings (occupancy)? tick
    
*   How does cancellation policy affect pricing strategy? tick
    

### Occupancy & Booking Patterns

*   What's the average occupancy rate across the dataset? tick
    
*   Do flexible cancellation policies lead to higher occupancy? tick
    
*   How does minimum night requirement affect booking frequency? tick
    
*   Which listing types have the highest occupancy rates? tick
    
*   Are there seasonal trends in your data (if dates are available)? unfortunately, no dates available
    

### Host Strategy & Management

*   Do professionally managed listings have better reviews/ratings? NO Tick
    
*   What percentage of listings are professionally managed vs. individual? tick
    
*   Do professional hosts charge more or less than individual hosts? tick
    
*   Is there a relationship between professional management and cancellation strictness? done
    
*   How many listings per host on average? done
    

### Geographic/Neighborhood Patterns

*   Are there neighborhood clusters with similar pricing? done
    
*   Do certain neighborhoods favor specific room types? done
    
*   Which areas have the highest revenue potential? done
    

### Quality & Features

*   How many amenities do high-revenue listings typically have?
    
*   Do listings with cover photos perform better? Done
    
*   What's the relationship between listing name length and bookings?
    

### Market Opportunities

*   What's underrepresented in the market (low supply, high demand)?
    
*   Are there price gaps where new listings could compete?
    
*   Which listing types have declining revenue trends?

---

## 📉 Visualizations
You may include screenshots or links to charts:
- Time series plots  
- Bar/line charts  
- Heatmaps  
- Forecast outputs  
- Pie Charts

---

## 🤖 Models (Optional)
If applicable:
- Model type  
- Metrics used  
- Performance summary  

---

## 🚀 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/username/repo-name.git
cd repo-name

# Create environment
conda create -n analysis_env python=3.10
conda activate analysis_env

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook
