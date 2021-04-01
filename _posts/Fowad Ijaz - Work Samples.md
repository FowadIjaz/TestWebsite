
# Fowad Ijaz, MBA
## Generating insights through data




---


#[Stock Price Prediction Using Machine Learning Algorithms](https://colab.research.google.com/drive/1anuR5juj9BDKQ5s359DFmdtyYGonBAVM?usp=sharing)

Working in a team, we predicted the stock prices of all stocks in the S&P500 index for a period of 4 years. Using pandas and numpy for basic data extraction and cleaning, matplotlib for exploratory data analysis, and the scikit-learn package for the Machine Learning algorithms, we applied 3 algorithms and assessed the efficacy of each. 

The model with the highest accuracy was the GradientBoost Regressor (below).


```

#plots
for i,j in enumerate([3, 38, 154, 207]):
  Chart3 = Chart2[Chart2.Name == j]
  plt.figure(figsize=(14,5) , dpi= 100, facecolor='w', edgecolor='k')
  plt.plot(Chart3['Y_test'], label='Actual Closing Price')
  plt.plot(Chart3['Y_test_GBR'], color = "red", label='Predicted Closing Price - Gradient Boost Regressor')
  plt.legend()
  plt.title(f"Actual vs Predicted Closing Stock Prices for {stocknames[i]} - Gradient Boost Regressor")
# 3, 38, 154, 207 Corresponds to the 4 sample stock names: AAPL, AMZN, EBAY, GOOGL
```


![png](Fowad%20Ijaz%20-%20Work%20Samples_files/Fowad%20Ijaz%20-%20Work%20Samples_2_0.png)



![png](Fowad%20Ijaz%20-%20Work%20Samples_files/Fowad%20Ijaz%20-%20Work%20Samples_2_1.png)



![png](Fowad%20Ijaz%20-%20Work%20Samples_files/Fowad%20Ijaz%20-%20Work%20Samples_2_2.png)



![png](Fowad%20Ijaz%20-%20Work%20Samples_files/Fowad%20Ijaz%20-%20Work%20Samples_2_3.png)




---


# [BCG Virtual Experience Program](https://colab.research.google.com/drive/1oYsP8d9vrOI-tPNrW4Hg_TVhAaacoc52?usp=sharing)
In winter 2020, I participated in the open access BCG Virtual Experience Program with Forage. The goal was to go through all steps of the data science process using real customer data from a telecommunications company. I undertook the following major steps to present an actionable solution to the company:



1.   Business Understanding & Problem Framing
2.   Exploratory Data Analaysis & Data Cleaning
3.   Feature Engineering
4.   Modeling & Evaluation
5.   Insights & Recommendation









---


# [Image Recognition Using Neural Networks](https://colab.research.google.com/drive/1uUBmzj8k4SvziSl5Ngk3bRv-7hNeQfZi?usp=sharing)

Using Tensorflow to train the dataset, I created a basic image recognition algorithm which was able to correctly classify images with an 88% accuracy. 


```
# Pre-process the source data (clamp pixel values between 0 - 1)
train_images = train_images / 255.0
test_images = test_images / 255.0
```


```
# Inspect pre-processed data (pixel values between 0 - 1)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```


![png](Fowad%20Ijaz%20-%20Work%20Samples_files/Fowad%20Ijaz%20-%20Work%20Samples_6_0.png)



```
cm= tf.math.confusion_matrix(labels = test_labels, predictions = estimates)

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt ='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')

```




    Text(69.0, 0.5, 'Actual')




![png](Fowad%20Ijaz%20-%20Work%20Samples_files/Fowad%20Ijaz%20-%20Work%20Samples_7_1.png)




---


# [Creating Heatmap visualizations using Bokeh](https://colab.research.google.com/drive/1Oh9dpK_G9XXEU5L6-NEdVwaUs8-mGqGJ?usp=sharing)

I created a heatmap of Ontario that could be broken into segments based on Canadian Census data. The data was then mapped onto boundary files provided by the Government of Canada on the StatsCan website. The results were graphed using the Bokeh package in Python.

### Results

*Confidence in Big Business*

The plot below shows that confidence in big business is not higher than 40% anywhere, but is consistently lower in Southern Ontario.


```
df2.plot_bokeh(simplify_shapes=1000,
                  category="Confidence", 
                  colormap="Spectral", 
                  hovertool_columns=["CFSAUID","Confidence"])
```










  <div class="bk-root" id="5259e734-03c1-48dc-92e4-fa7ae295d3c9" data-root-id="1801"></div>








<div style="display: table;"><div style="display: table-row;"><div style="display: table-cell;"><b title="bokeh.plotting.figure.Figure">Figure</b>(</div><div style="display: table-cell;">id&nbsp;=&nbsp;'1801', <span id="1939" style="cursor: pointer;">&hellip;)</span></div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">above&nbsp;=&nbsp;[],</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">align&nbsp;=&nbsp;'start',</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">aspect_ratio&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">aspect_scale&nbsp;=&nbsp;1,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">background&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">background_fill_alpha&nbsp;=&nbsp;1.0,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">background_fill_color&nbsp;=&nbsp;'#ffffff',</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">below&nbsp;=&nbsp;[MercatorAxis(id='1812', ...)],</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">border_fill_alpha&nbsp;=&nbsp;1.0,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">border_fill_color&nbsp;=&nbsp;'#ffffff',</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">center&nbsp;=&nbsp;[Grid(id='1819', ...), Grid(id='1827', ...), Legend(id='1863', ...)],</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">css_classes&nbsp;=&nbsp;[],</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">disabled&nbsp;=&nbsp;False,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">extra_x_ranges&nbsp;=&nbsp;{},</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">extra_y_ranges&nbsp;=&nbsp;{},</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">frame_height&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">frame_width&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">height&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">height_policy&nbsp;=&nbsp;'auto',</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">hidpi&nbsp;=&nbsp;True,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_event_callbacks&nbsp;=&nbsp;{},</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_property_callbacks&nbsp;=&nbsp;{},</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">left&nbsp;=&nbsp;[MercatorAxis(id='1820', ...)],</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_factor&nbsp;=&nbsp;10,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_interval&nbsp;=&nbsp;300,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_threshold&nbsp;=&nbsp;2000,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_timeout&nbsp;=&nbsp;500,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">margin&nbsp;=&nbsp;(0, 0, 0, 0),</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">match_aspect&nbsp;=&nbsp;False,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">max_height&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">max_width&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border&nbsp;=&nbsp;5,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_bottom&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_left&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_right&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_top&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_height&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_width&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">name&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_alpha&nbsp;=&nbsp;1.0,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_cap&nbsp;=&nbsp;'butt',</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_color&nbsp;=&nbsp;'#e5e5e5',</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_dash&nbsp;=&nbsp;[],</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_dash_offset&nbsp;=&nbsp;0,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_join&nbsp;=&nbsp;'bevel',</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_width&nbsp;=&nbsp;1,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">output_backend&nbsp;=&nbsp;'webgl',</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">plot_height&nbsp;=&nbsp;400,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">plot_width&nbsp;=&nbsp;600,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">renderers&nbsp;=&nbsp;[TileRenderer(id='1844', ...), GlyphRenderer(id='1853', ...)],</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">reset_policy&nbsp;=&nbsp;'standard',</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">right&nbsp;=&nbsp;[ColorBar(id='1867', ...)],</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">sizing_mode&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">subscribed_events&nbsp;=&nbsp;[],</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">tags&nbsp;=&nbsp;[],</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">title&nbsp;=&nbsp;Title(id='1802', ...),</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">title_location&nbsp;=&nbsp;'above',</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar&nbsp;=&nbsp;Toolbar(id='1835', ...),</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar_location&nbsp;=&nbsp;'right',</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar_sticky&nbsp;=&nbsp;True,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">visible&nbsp;=&nbsp;True,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">width&nbsp;=&nbsp;None,</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">width_policy&nbsp;=&nbsp;'auto',</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_range&nbsp;=&nbsp;DataRange1d(id='1804', ...),</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_scale&nbsp;=&nbsp;LinearScale(id='1808', ...),</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_range&nbsp;=&nbsp;DataRange1d(id='1806', ...),</div></div><div class="1938" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_scale&nbsp;=&nbsp;LinearScale(id='1810', ...))</div></div></div>
<script>
(function() {
  var expanded = false;
  var ellipsis = document.getElementById("1939");
  ellipsis.addEventListener("click", function() {
    var rows = document.getElementsByClassName("1938");
    for (var i = 0; i < rows.length; i++) {
      var el = rows[i];
      el.style.display = expanded ? "none" : "table-row";
    }
    ellipsis.innerHTML = expanded ? "&hellip;)" : "&lsaquo;&lsaquo;&lsaquo;";
    expanded = !expanded;
  });
})();
</script>




*Tech Enthusiasm*

Surprisingly, tech enthusiasm appears to be highest in Northern Ontario. This could be due to a number of factors including small sample size, or possibly because a remote population is able to understand the role that technology is able to play in enhancing people's quality of life. 

Another major hotspot with high tech enthusiasm is, of course, Toronto. When zoomed in, we can see that it is consistently higher than its surrounding regions. 


```

df2.plot_bokeh(simplify_shapes=1000,
                  category="Tech.Enthu.", 
                  colormap="Spectral", 
                  hovertool_columns=["CFSAUID","Tech.Enthu."])

```










  <div class="bk-root" id="92c5bcc1-bd22-41d3-a3c9-6dd975bca30d" data-root-id="1941"></div>








<div style="display: table;"><div style="display: table-row;"><div style="display: table-cell;"><b title="bokeh.plotting.figure.Figure">Figure</b>(</div><div style="display: table-cell;">id&nbsp;=&nbsp;'1941', <span id="2089" style="cursor: pointer;">&hellip;)</span></div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">above&nbsp;=&nbsp;[],</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">align&nbsp;=&nbsp;'start',</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">aspect_ratio&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">aspect_scale&nbsp;=&nbsp;1,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">background&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">background_fill_alpha&nbsp;=&nbsp;1.0,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">background_fill_color&nbsp;=&nbsp;'#ffffff',</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">below&nbsp;=&nbsp;[MercatorAxis(id='1952', ...)],</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">border_fill_alpha&nbsp;=&nbsp;1.0,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">border_fill_color&nbsp;=&nbsp;'#ffffff',</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">center&nbsp;=&nbsp;[Grid(id='1959', ...), Grid(id='1967', ...), Legend(id='2003', ...)],</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">css_classes&nbsp;=&nbsp;[],</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">disabled&nbsp;=&nbsp;False,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">extra_x_ranges&nbsp;=&nbsp;{},</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">extra_y_ranges&nbsp;=&nbsp;{},</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">frame_height&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">frame_width&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">height&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">height_policy&nbsp;=&nbsp;'auto',</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">hidpi&nbsp;=&nbsp;True,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_event_callbacks&nbsp;=&nbsp;{},</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_property_callbacks&nbsp;=&nbsp;{},</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">left&nbsp;=&nbsp;[MercatorAxis(id='1960', ...)],</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_factor&nbsp;=&nbsp;10,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_interval&nbsp;=&nbsp;300,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_threshold&nbsp;=&nbsp;2000,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_timeout&nbsp;=&nbsp;500,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">margin&nbsp;=&nbsp;(0, 0, 0, 0),</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">match_aspect&nbsp;=&nbsp;False,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">max_height&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">max_width&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border&nbsp;=&nbsp;5,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_bottom&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_left&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_right&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_top&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_height&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_width&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">name&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_alpha&nbsp;=&nbsp;1.0,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_cap&nbsp;=&nbsp;'butt',</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_color&nbsp;=&nbsp;'#e5e5e5',</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_dash&nbsp;=&nbsp;[],</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_dash_offset&nbsp;=&nbsp;0,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_join&nbsp;=&nbsp;'bevel',</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_width&nbsp;=&nbsp;1,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">output_backend&nbsp;=&nbsp;'webgl',</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">plot_height&nbsp;=&nbsp;400,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">plot_width&nbsp;=&nbsp;600,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">renderers&nbsp;=&nbsp;[TileRenderer(id='1984', ...), GlyphRenderer(id='1993', ...)],</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">reset_policy&nbsp;=&nbsp;'standard',</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">right&nbsp;=&nbsp;[ColorBar(id='2007', ...)],</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">sizing_mode&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">subscribed_events&nbsp;=&nbsp;[],</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">tags&nbsp;=&nbsp;[],</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">title&nbsp;=&nbsp;Title(id='1942', ...),</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">title_location&nbsp;=&nbsp;'above',</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar&nbsp;=&nbsp;Toolbar(id='1975', ...),</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar_location&nbsp;=&nbsp;'right',</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar_sticky&nbsp;=&nbsp;True,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">visible&nbsp;=&nbsp;True,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">width&nbsp;=&nbsp;None,</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">width_policy&nbsp;=&nbsp;'auto',</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_range&nbsp;=&nbsp;DataRange1d(id='1944', ...),</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_scale&nbsp;=&nbsp;LinearScale(id='1948', ...),</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_range&nbsp;=&nbsp;DataRange1d(id='1946', ...),</div></div><div class="2088" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_scale&nbsp;=&nbsp;LinearScale(id='1950', ...))</div></div></div>
<script>
(function() {
  var expanded = false;
  var ellipsis = document.getElementById("2089");
  ellipsis.addEventListener("click", function() {
    var rows = document.getElementsByClassName("2088");
    for (var i = 0; i < rows.length; i++) {
      var el = rows[i];
      el.style.display = expanded ? "none" : "table-row";
    }
    ellipsis.innerHTML = expanded ? "&hellip;)" : "&lsaquo;&lsaquo;&lsaquo;";
    expanded = !expanded;
  });
})();
</script>



