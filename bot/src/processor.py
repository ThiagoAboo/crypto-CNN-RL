import mplfinance as mpf
import io
import numpy as np
from PIL import Image
import bot.config as config

class ImageProcessor:
    def __init__(self):
        self.size = config.IMG_SIZE
        if config.DEBUG: 
            print(f"[TRACE] ImageProcessor configurado: {self.size}px, {config.IMG_DPI} DPI.")

    def dataframe_to_numpy(self, df_window):
        buf = io.BytesIO()
        fig_scale = self.size[0] / config.IMG_DPI # Usando o primeiro valor da tupla

        mpf.plot(
            df_window, 
            type='candle', 
            style='charles', 
            axisoff=True, 
            savefig=dict(fname=buf, dpi=config.IMG_DPI, bbox_inches='tight', pad_inches=0),
            figsize=(fig_scale, fig_scale)
        )
        
        buf.seek(0)
        img = Image.open(buf).convert('L').resize(self.size)
        # Garantindo float32 e cópia limpa para o NumPy 2.0
        img_array = np.array(img).astype(np.float32) / 255.0  
        buf.close()
        
        # Retorna com a dimensão do canal (1, H, W)
        return np.expand_dims(img_array, axis=0).copy()
