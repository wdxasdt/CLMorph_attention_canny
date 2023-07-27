img_index = int(index % len(self.moving_path))

moving_img = self.whitening(self.readVol(self.moving_path[img_index]))
fixed_img = self.fixed_img_whiten[self.moving_fixed[self.moving_path[img_index]]]

