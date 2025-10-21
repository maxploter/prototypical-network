import unittest
import torch
from .episode_sampler import EpisodeSampler


class TestEpisodeSampler(unittest.TestCase):

    def setUp(self):
        self.labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
        self.classes_per_iteration = 2
        self.samples_per_class = 3
        self.number_of_iterations = 5

    def test_initialization(self):
        sampler = EpisodeSampler(
            labels=torch.tensor([0, 2, 2, 1, 2, 1, 0, 1, 1, 2, 0, 1]),
            classes_per_iteration=2,
            samples_per_class=3,
            number_of_iterations=5
        )

        self.assertEqual(len(sampler.unique_labels), 3)
        self.assertEqual(sampler.indeces_per_class.shape, (3, 5))
        self.assertEqual(sampler.indeces_per_class[0].tolist(), [0, 6, 10, 0, 0])  # Class 0 indices
        self.assertEqual(sampler.indeces_per_class[1].tolist(), [3, 5, 7, 8, 11])  # Class 1 indices
        self.assertEqual(sampler.indeces_per_class[2].tolist(), [1, 2, 4, 9, 0])   # Class 2 indices
        self.assertEqual(sampler.label_counts.tolist(), [3, 5, 4])
        self.assertEqual(sampler.classes_per_iteration, 2)
        self.assertEqual(sampler.samples_per_class, 3)
        self.assertEqual(sampler.number_of_iterations, 5)

    def test_len(self):
        """Test that __len__ returns the correct number of iterations."""
        sampler = EpisodeSampler(
            labels=self.labels,
            classes_per_iteration=self.classes_per_iteration,
            samples_per_class=self.samples_per_class,
            number_of_iterations=self.number_of_iterations
        )

        self.assertEqual(len(sampler), self.number_of_iterations)

    def test_iter_yields_correct_number_of_episodes(self):
        """Test that __iter__ yields the correct number of episodes."""
        sampler = EpisodeSampler(
            labels=torch.tensor([0, 2, 2, 1, 2, 1, 0, 1, 1, 2, 0, 1]),
            classes_per_iteration=2,
            samples_per_class=3,
            number_of_iterations=5
        )

        episodes = list(sampler)
        self.assertEqual(len(episodes), self.number_of_iterations)

    def test_episode_size(self):
        """Test that each episode has the correct number of samples."""
        sampler = EpisodeSampler(
            labels=self.labels,
            classes_per_iteration=self.classes_per_iteration,
            samples_per_class=self.samples_per_class,
            number_of_iterations=self.number_of_iterations
        )

        for episode_indices in sampler:
            expected_size = self.classes_per_iteration * self.samples_per_class
            self.assertEqual(len(episode_indices), expected_size)

    def test_indices_are_valid(self):
        """Test that all sampled indices are valid."""
        sampler = EpisodeSampler(
            labels=self.labels,
            classes_per_iteration=self.classes_per_iteration,
            samples_per_class=self.samples_per_class,
            number_of_iterations=self.number_of_iterations
        )

        for episode_indices in sampler:
            for idx in episode_indices:
                # Check that index is within valid range
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(self.labels))

    def test_samples_come_from_correct_classes(self):
        """Test that samples in each episode come from the expected number of classes."""
        sampler = EpisodeSampler(
            labels=self.labels,
            classes_per_iteration=self.classes_per_iteration,
            samples_per_class=self.samples_per_class,
            number_of_iterations=self.number_of_iterations
        )

        for episode_indices in sampler:
            # Get the labels of sampled indices
            sampled_labels = self.labels[episode_indices]
            # Count unique classes
            unique_classes = torch.unique(sampled_labels)

            # Should have exactly classes_per_iteration unique classes
            self.assertEqual(len(unique_classes), self.classes_per_iteration)

    def test_multiple_iterations(self):
        """Test that the sampler can be iterated multiple times."""
        sampler = EpisodeSampler(
            labels=self.labels,
            classes_per_iteration=self.classes_per_iteration,
            samples_per_class=self.samples_per_class,
            number_of_iterations=3
        )

        # First iteration
        first_episodes = list(sampler)
        self.assertEqual(len(first_episodes), 3)

        # Second iteration
        second_episodes = list(sampler)
        self.assertEqual(len(second_episodes), 3)

    def test_single_class_dataset(self):
        """Test with a dataset containing only one class."""
        single_class_labels = torch.tensor([0, 0, 0, 0, 0])

        sampler = EpisodeSampler(
            labels=single_class_labels,
            classes_per_iteration=1,
            samples_per_class=2,
            number_of_iterations=2
        )

        for episode_indices in sampler:
            self.assertEqual(len(episode_indices), 2)
            # All should be from class 0
            sampled_labels = single_class_labels[episode_indices]
            self.assertTrue(torch.all(sampled_labels == 0))