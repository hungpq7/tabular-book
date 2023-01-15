#!/usr/bin/env python
# coding: utf-8

# # Plotly: Interactive Visualization

# ## 1. Introduction
# [Plotly](https://plotly.com/python/) is an Python library interactive, open-source plotting library that supports over 40 unique chart types covering a wide range of statistical, financial, geographic, scientific, and 3-dimensional use-cases.
# 
# Plotly has two important submodules, <code style='font-size:13px;'>plotly.express</code> and <code style='font-size:13px;'>plotly.graph_objects</code>, but this topic focuses on <code style='font-size:13px;'>plotly.express</code> only. It allows quickly creating more than 30 types of charts, each made only in a single function call.

# ## 1. Introduction
# [Plotly](https://plotly.com/python/) is an Python library interactive, open-source plotting library that supports over 40 unique chart types covering a wide range of statistical, financial, geographic, scientific, and 3-dimensional use-cases.
# 
# Plotly has two important submodules, `plotly.express` and `plotly.graph_objects`, but this topic focuses on `plotly.express` only. It allows quickly creating more than 30 types of charts, each made only in a single function call.

# ### Useful scripts
# 
# To display interactive Plotly charts in Jupyter Lab, it is required to install an extension.
# ```python
# jupyter labextension install jupyterlab-plotly@4.12.0
# ```
# 
# To export a chart to a <code style='font-size:13px;'>.html</code> file, use the script below in which <code style='font-size:13px;'>fig</code> is the name of the graph object.
# ```python
# import plotly
# plotly.offline.plot(fig, filename='My interactive figure.html')
# ```

# ## 2. Basic charts

# ### 2.1. Pie Chart

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


dfTip = px.data.tips()
dfTip.head()


# In[3]:


fig = px.pie(dfTip, names='day', values='tip', template='plotly')
fig.show()


# #### Donut Chart

# In[4]:


dfTip = px.data.tips()
dfTip.head()


# In[5]:


fig = px.pie(dfTip, names='day', values='tip', hole=.5)
fig.show()


# ### 2.2. Bar Chart

# In[6]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# #### Grouping and stacking

# In[7]:


dfSales = pd.read_csv('data/supermarket_sales.csv')
dfSales.head()


# In[8]:


fig = px.bar(
    dfSales.groupby(['payment', 'product_line']).sum().reset_index(),
    x='product_line',
    y='profit',
    color='payment',
    barmode='group',
    color_discrete_sequence=['#34495e', 'indianred', 'teal']
)
fig.show()


# In[9]:


fig = px.bar(
    dfSales.groupby(['payment', 'product_line']).sum().reset_index(),
    y='product_line',
    x='profit',
    color='payment',
    barmode='stack',
    color_discrete_sequence=['#34495e', 'indianred', 'teal']
)
fig.show()


# #### Facet grid

# In[10]:


dfSales = pd.read_csv('data/supermarket_sales.csv')
dfSales.head()


# In[11]:


fig = px.bar(
    dfSales.groupby(['payment', 'gender', 'city', 'customer_type']).sum().reset_index(),
    y='gender',
    x='profit',
    color='payment',
    facet_row='city',
    facet_col='customer_type',
    color_discrete_sequence=['#34495e', 'indianred', 'teal'],
    barmode='group'
)
fig.show()


# In[12]:


fig = px.bar(
    dfSales.groupby(['payment', 'gender', 'product_line']).sum().reset_index(),
    x='gender',
    y='profit',
    color='payment',
    facet_col='product_line',
    facet_col_wrap=3,
    color_discrete_sequence=['#34495e', 'indianred', 'teal'],
    barmode='group'
)
fig.show()


# #### Bar Chart with slider

# In[13]:


countries = ['GNQ', 'BWA', 'GAB', 'PRI', 'NZL', 'AUS', 'CAN', 'CHE', 'HKG', 'IRL', 'USA', 'SGP', 'KWT', 'NOR']
dfWorld = pd.read_csv('data/gapminder.csv')
dfWorld = dfWorld[dfWorld.iso_alpha.isin(countries)]
dfWorld = dfWorld.reset_index()
dfWorld.head()


# In[14]:


fig = px.bar(
    dfWorld,
    x='country', y='gdpPercap', color='continent',
    animation_frame='year',
    log_y=True,
    range_y=[100, 200000],
    color_discrete_sequence=['#34495e', 'indianred', 'teal', '#F39C12', 'steelblue']
)
fig.show()


# #### Dot Plot

# In[15]:


fig = px.scatter(
    dfSales.groupby(['payment', 'product_line']).sum().reset_index(),
    y='product_line',
    x='profit',
    color='payment',
#     color_discrete_sequence=['#34495e', 'indianred', 'teal']
)
fig.show()


# ### 2.3. Line Chart

# In[16]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[17]:


dfWorld = pd.read_csv('data/gapminder.csv')
dfWorld = dfWorld.reset_index()
dfWorld.head()


# In[18]:


fig = px.line(
    dfWorld.groupby(['continent', 'year']).mean().reset_index(),
    x='year',
    y='gdpPercap',
    color='continent'
)
fig.show()


# #### Line Chart with slider

# In[19]:


dfApple = pd.read_csv('data/finance_charts_apple.csv')
dfApple.columns = [col.replace("AAPL.", '') for col in dfApple.columns]
dfApple.head()


# In[20]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=list(dfApple.Date), y=list(dfApple.High)))
fig.update_layout(title_text='Time series with range slider and selectors')

# range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1y', step='year', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible=True),
        type='date'
    )
)

fig.show()


# ### 2.4. Scatter Plot

# In[21]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# #### Dot Plot

# In[22]:


dfSales = pd.read_csv('data/supermarket_sales.csv')
dfSales.head()


# In[23]:


fig = px.scatter(
    dfSales.groupby(['payment', 'product_line']).sum().reset_index(),
    y='product_line',
    x='profit',
    color='payment',
)
fig.show()


# #### 3D Scatter Plot

# In[24]:


dfIris = px.data.iris()
dfIris.head()


# In[25]:


fig = px.scatter_3d(
    dfIris,
    x='sepal_length',
    y='sepal_width',
    z='petal_width',
    color='species',
    template='seaborn'
)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()


# #### Scatter Plot with slider

# In[26]:


fig = px.scatter(
    px.data.gapminder(), 
    x='gdpPercap', y='lifeExp',
    size='pop', color='continent',
    animation_frame='year',
    hover_name='country',
    log_x=True,
    size_max=100,
    range_x=[100,100000],
    range_y=[25,100],
    template='seaborn'
)

fig.show()


# ### 2.5. Radar Chart

# In[27]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[28]:


dfKnowledge = pd.DataFrame({
    'Subject': ['Statistics', 'Econometrics', 'Linear Algebra', 'Calculus', 'Machine Learning', 'Visualization'],
    'Hưng': [10, 8, 9, 10, 8, 10],
    'Linh': [9, 10, 5, 7, 4, 7],
}).melt(id_vars='Subject', var_name='Name', value_name='Score')

fig = px.line_polar(
    dfKnowledge, r='Score', theta='Subject', color='Name',
    line_close=True,
    color_discrete_sequence=['teal', 'brown'],
    template='seaborn', title='Data Analytics Knowledge'
)
fig.show()


# In[29]:


dfSkills = pd.DataFrame({
    'Skills': ['Excel', 'SQL', 'Python', 'R', 'Tableau', 'PowerPoint'],
    'Score': [10, 10, 10, 9, 7, 10],
    'Name': ['Hung']*6
})

fig = px.line_polar(
    dfSkills, r='Score', theta='Skills', color='Name',
    line_close=True,
    color_discrete_sequence=['#34495E'],
    template='seaborn', title='Data Analytics Skills'
)
fig.update_traces(fill='toself')
fig.show()


# ### 2.6. Gantt Chart

# In[2]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[8]:


data = [
    ['Phase A', 'Task A1', '2020-06-01', '2020-06-15'],
    ['Phase A', 'Task A2', '2020-06-15', '2020-06-22'],
    ['Phase A', 'Task A3', '2020-06-22', '2020-06-30'],
    ['Phase B', 'Task B1', '2020-07-01', '2020-07-20'],
    ['Phase B', 'Task B2', '2020-07-15', '2020-07-31'],
    ['Phase B', 'Task B3', '2020-08-01', '2020-08-20'],
    ['Phase B', 'Task B1', '2020-08-01', '2020-08-05'],
    ['Phase B', 'Task B4', '2020-08-12', '2020-08-21'],
    ['Phase B', 'Task B5', '2020-08-15', '2020-09-05'],
    ['Phase B', 'Task B2', '2020-08-20', '2020-08-31'],
    ['Phase C', 'Task C1', '2020-09-01', '2020-09-27'],
    ['Phase C', 'Task C2', '2020-09-25', '2020-10-05'],
    ['Phase C', 'Task C3', '2020-10-05', '2020-10-15'],
    ['Phase C', 'Task C4', '2020-10-15', '2020-10-30'],
]

columns = ['phase', 'task', 'start', 'end']

dfTimeline = pd.DataFrame(data=data, columns=columns)
dfTimeline.head(3)


# In[9]:


fig = px.timeline(
    dfTimeline,
    x_start='start',
    x_end='end',
    y='task',
    color='phase'
)
fig.update_yaxes(autorange='reversed')
fig.show()


# ### 2.7. Violin Plot

# In[33]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# In[34]:


dfTip = sns.load_dataset('tips')
dfTip.head()


# In[35]:


fig = px.violin(
    dfTip,
    x='smoker',
    y='total_bill',
    color='sex',
    box=True,
    points='all'
)
fig.show()


# ## 3. Advanced charts

# ### 3.1. Sunburst Chart

# In[36]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[37]:


dfTip = px.data.tips()
dfTip.head()


# In[38]:


fig = px.sunburst(dfTip, path=['day', 'time', 'sex'], values='total_bill')
fig.show()


# ### 3.2. Tree Map

# In[39]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[40]:


dfWorld = pd.read_csv('data/gapminder.csv')
dfWorld['world'] = 'World'
dfWorld = dfWorld[dfWorld.year==2007]
dfWorld.head()


# In[41]:


fig = px.treemap(
    dfWorld,
    values='pop',
    path=['country'],
    color='continent'
)
fig.show()


# In[42]:


fig = px.treemap(
    dfWorld,
    values='pop',
    path=['world', 'continent', 'country'],
)
fig.show()


# In[43]:


fig = px.treemap(
    dfWorld,
    values='pop',
    path=['world', 'continent', 'country'],
    color='lifeExp',
    color_continuous_scale='RdBu',
    color_continuous_midpoint=np.average(dfWorld['lifeExp'], weights=dfWorld['pop'])
)
fig.show()


# ### 3.3. Parallel Set Diagram

# In[44]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[45]:


dfTip = px.data.tips()
dfTip.head()


# In[46]:


fig = px.parallel_categories(
    dfTip,
    dimensions=['sex', 'smoker', 'day', 'time'],
    color='size',
    color_continuous_scale='OrRd'
)
fig.show()


# ### 3.4. Parallel Coordinates Plot

# In[47]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[48]:


dfIris = px.data.iris()
dfIris.head()


# In[49]:


fig = px.parallel_coordinates(
    dfIris,
    color='species_id',
    dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    color_continuous_scale='Temps'
)
fig


# ## 4. Maps
# Useful map styles that can be controlled using the
# <code style='font-size:13px;'><a href='https://plotly.com/python/mapbox-layers/'>mapbox_style</a></code>
# parameter:
# - <code style='font-size:13px;'>white-bg</code>
# - <code style='font-size:13px;'>open-street-map</code>
# - <code style='font-size:13px;'>carto-positron</code>
# - <code style='font-size:13px;'>carto-darkmatter</code>
# - <code style='font-size:13px;'>stamen-terrain</code>
# - <code style='font-size:13px;'>stamen-toner</code>
# - <code style='font-size:13px;'>stamen-watercolor</code>
# 
# Plotly support other style as well:
# <code style='font-size:13px;'>basic</code>
# <code style='font-size:13px;'>streets</code>
# <code style='font-size:13px;'>outdoors</code>
# <code style='font-size:13px;'>light</code>
# <code style='font-size:13px;'>dark</code>
# <code style='font-size:13px;'>satellite</code>
# <code style='font-size:13px;'>satellite-streets</code>,
# but they require a Mapbox access token.

# ### 4.1. Density Heat Map
# [Density Heat Maps](https://plotly.com/python/mapbox-density-heatmaps/) show the magnitude of a phenomenon and where it is located. The
# <code style='font-size:13px;'><a href='https://plotly.com/python-api-reference/generated/plotly.express.density_mapbox'>px.density_mapbox()</a></code>
# function has the notable parameters as follows:
# - <code style='font-size:13px;'>lat</code>: the latitude
# - <code style='font-size:13px;'>lon</code>: the longitude
# - <code style='font-size:13px;'>z</code>: the magnitude or the size

# In[50]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[51]:


dfEarthquake = pd.read_csv('data/earthquakes.csv')
dfEarthquake.head()


# In[52]:


fig = px.density_mapbox(
    dfEarthquake,
    lat='Latitude',
    lon='Longitude',
    z='Magnitude',
    radius=10,
    center={'lat':0, 'lon':180},
    zoom=1,
    color_continuous_scale='Tealgrn',
    mapbox_style="carto-positron")

fig.show()


# ### 4.2. Bubble Map
# [Bubble Maps](https://plotly.com/python/bubble-maps/), implemented via
# <code style='font-size:13px;'><a href='https://plotly.github.io/plotly.py-docs/generated/plotly.express.scatter_mapbox.html'>px.scatter_mapbox()</a></code>,
# takes the same input as Density Heat Map.

# In[53]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[54]:


dfCarshare = px.data.carshare()
dfCarshare.head()


# In[55]:


fig = px.scatter_mapbox(
    dfCarshare,
    lat='centroid_lat', lon='centroid_lon',
    color='peak_hour', size='car_hours',
    color_continuous_scale=px.colors.cyclical.IceFire,
    size_max=15, zoom=10,
    mapbox_style='carto-positron')

fig.show()


# ### 4.3. Choropleth Map
# [Choropleth Maps](https://plotly.com/python/choropleth-maps/) visualize a quantitative variable using colored polygons. The
# <code style='font-size:13px;'><a href='https://plotly.github.io/plotly.py-docs/generated/plotly.express.choropleth_mapbox.html'>px.choropleth_mapbox()</a></code>
# function has the notable parameters as follows:
# - <code style='font-size:13px;'>geojson</code>: dictionary-styled geometry information that must contain the latitudes and the longitudes of polygon vertexes.
# - <code style='font-size:13px;'>locations</code>: names of colored regions from <code style='font-size:13px;'>data_frame</code>.
# - <code style='font-size:13px;'>featureidkey</code>: to specify the path used in <code style='font-size:13px;'>geojson</code> that match the value of <code style='font-size:13px;'>locations</code>.

# In[56]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[57]:


dfElection = px.data.election()
geojson = px.data.election_geojson()
dfElection.head()


# In[58]:


geojson['features'][0]['properties']


# In[59]:


dfElection.district[2]


# In[60]:


fig = px.choropleth_mapbox(
    dfElection,
    geojson=geojson,
    color="winner",
    locations="district",
    featureidkey="properties.district",
    center={"lat": 45.5517, "lon": -73.707},
    zoom=9,
    mapbox_style='carto-positron')

fig.show()


# ---
# *&#9829; By Quang Hung x Thuy Linh &#9829;*
