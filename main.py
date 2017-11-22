from models.cyclegan import CycleGAN

def main():
    cyclegan = CycleGAN()
    cyclegan.train()

if __name__ == '__main__':
    main()

