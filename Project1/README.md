<pre><code>
conda create --name p1
source activate p1
python main.py
</code></pre>

Create Env
<code>conda env create -f /path/to/environment.yml</code>

Update Env
<code>conda env update -f /path/to/environment.yml</code>

Update environment.yml file
<code>conda env export  -f /path/to/environment.yml --no-builds</code>
