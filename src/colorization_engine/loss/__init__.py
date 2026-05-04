from colorization_engine.loss.base_loss import BaseLoss

from colorization_engine.loss.colorization import ColorizationLoss

# import importlib
# import pkgutil

# def import_all_loss():
#     import colorization_engine.loss as loss_pkg

#     for _, module_name, _ in pkgutil.iter_modules(loss_pkg.__path__):
#         importlib.import_module(f"{loss_pkg.__name__}.{module_name}")