# Text to handwriting !!!
### Get given text written on a ruled page automatically
**Its time for your laptop to write assignments for you!** \
**Click to see [Example input](https://github.com/pnshiralkar/text-to-handwriting/blob/master/Example/input.txt) and [Example output](https://github.com/pnshiralkar/text-to-handwriting/blob/master/Example/handwritten.pdf) .**

Implementation of handwriting generation with use of recurrent neural networks in tensorflow. Based on Alex Graves paper (https://arxiv.org/abs/1308.0850). \
This project uses pretrained model and some implementation based on the paper from [this](https://github.com/theSage21/handwriting-generation) repo. 

## Install and Use
* Download zip or clone this repo and cd into the repo folder
* Install dependencies : `pip install -r requirements.txt` OR `pip3 install -r requirements.txt`
* **Run and Use :**
   * `python handwrite.py --text "Some text with minimum 50 characters" <optional arguments>`
   * `python handwrite.py --text-file /path/to/input/text.file <optional arguments>`
* Optional Arguments :
    * `--style` : Style of handwriting (0 to 7, defaults to 0)
    * `--bias` : Bias in handwriting. More bias is more unclear handwriting (0.00 to 1.00 , defaults to 0.9)
    * `--color` : Color of handwriting in RGB format ,defaults to 0,0,150 (ballpen blue)
    * `--output` : Path to output pdf file (E.g. ~/assignments/ads1.pdf), defaults to ./handwritten.pdf
    * For more information on usage, run `python handwrite.py -h`
    
### Works the best with multiple pages and long text!
    
## Additional Information :
* **Additional Outputs:** The pages folder stores the handwritten pages in .jpg and .png (transparent bg) format
* **Modification:** To modify, see generate.py file
* **Train model:** To modify, see train.py file (Refer [this](https://github.com/theSage21/handwriting-generation) repo for more)

More Info
---------

[The paper](http://arxiv.org/abs/1308.0850)  
[The man behind it all. Alex Graves](http://www.cs.toronto.edu/~graves/)  
[What I am using](https://github.com/theSage21/handwriting-generation)
