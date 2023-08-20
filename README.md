### Requirements
python == 3.10.8

torch == 1.13.1

einops == 0.6.0

numpy == 1.23.5

sklearn


### GANtrain
Setting hyper-parameters in ./GANtrain/cfg.py

```
cd GANtrain
python GANtrain.py

```
The generated data will be in ./results

### Decoding
Setting hyper-parameters in ./Decoding/cfg.py

#### without generated data:

    ```
    cd Decoding
    python main.py

    ```

#### with generated data:

    ```
    cd Decoding
    python main.py --generated_path '../results/generated_data.pth' 

    ```

If you want to change decoding models, use

```
python main.py --model_name ReSNN /

python main.py --model_name FCSNN /

python main.py --model_name LSTM

```


