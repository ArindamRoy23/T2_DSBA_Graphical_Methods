import argparse


class SVCDArgParser(object):
    """
    
    """
    def __init__(
            self
        ) -> None:
        self.parser = argparse.ArgumentParser(
            prog='SVCDSegmenter',
            description="""
                The program implements the "Spatially Varying Color Distribution for Interactive, Multilabel Image Segmentation" paper
                By AUTHOR1 and AUTHOR2.
                It requires a machine endowed with a cuda-capable GPU.
                This cli tool allow users to generate a segmentation mask for a given image
                using the algorithm derived in the mentioned paper, while also computing the dice score of the segmentation. 
                It saves the segmentation mask to the specified folder, with the same name as the target image
            """,
            epilog='Thank you very much for your attention'
        )
        self.parser.add_argument(
            '-i',
            '--target-image',
            type = str,
            help = "Input Image File",
            required = True
        )
        self.parser.add_argument(
            '-s',
            '--scribble',
            type = str,
            help = "Scribble File",
            required = True
        )
        self.parser.add_argument(
            '-o',
            '--output-folder',
            type = str,
            const = r"./plots/",
            default = r"./plots/",
            help = "Output File",
        )
        self.parser.add_argument(
            '-l'.
            '--lambda',
            type = float,
            const = 8e-4, 
            default = 8e-4, 
            help = "Lambda parameter for the model"
        )
        self.parser.add_argument(
            '-g'.
            '--gamma',
            type = float,
            const = 5e-0, 
            default = 5e-0, 
            help = "Gamma parameter for the model"
        )
        self.parser.add_argument(
            '-a'.
            '--alpha',
            type = float,
            const = 18e-1, 
            default = 18e-1,  
            help = "Alpha parameter for the model"
        )
        self.parser.add_argument(
            '-s'.
            '--sigma',
            type = float,
            const = 13e-1, 
            default = 13e-1,  
            help = "Sigma parameter for the model"
        )
        self.parser.add_argument(
            '-tp'.
            '--tau-primal',
            type = float, 
            const = 25e-2,
            default = 25e-2, 
            help = "Tau Primal parameter for the model"
        )
        self.parser.add_argument(
            '-td'.
            '--tau-dual',
            type = float,
            const = 5e-1,
            default = 5e-1, 
            help = "Tau Dual parameter for the model"
        )
        self.parser.add_argument(
            '-m'.
            '--max-iter',
            type = int,
            const = 1500,
            default = 1500, 
            help = "Maximum number of iterations for the model"
        )
        self.parser.add_argument(
            '-es'.
            '--early-stop',
            type = bool,
            const = False,
            default = False, 
            help = "Whether to use early stopping for the model"
        )
        self.parser.add_argument(
            '-tl'.
            '--tolerance',
            type = float, 
            const = 1e-5,
            default = 1e-5,
            help = "Improvement tolerance for the model"
        )
        self.parser.add_argument(
            '-ut'.
            '--use-tqdm',
            type = bool, 
            const = True,
            default = True,
            help = "Whether to use for the model"
        )
    
    def __call__(
            self
        ) -> argparse.Namespace:
        return self.parser.parse_args()
        