import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { Student } from '../../schemas/student.schema';

export interface CreateStudentDto {
  studentId: string;
  name: string;
  enrollmentPhotoCount?: number;
}

@Injectable()
export class StudentService {
  constructor(
    @InjectModel(Student.name) private studentModel: Model<Student>,
  ) {}

  async create(createStudentDto: CreateStudentDto): Promise<Student> {
    const student = new this.studentModel({
      ...createStudentDto,
      totalAttendanceCount: 0,
    });
    return student.save();
  }

  async findAll(): Promise<Student[]> {
    return this.studentModel.find().sort({ name: 1 }).exec();
  }

  async findById(studentId: string): Promise<Student | null> {
    return this.studentModel.findOne({ studentId }).exec();
  }

  async update(
    studentId: string,
    updateData: Partial<Student>,
  ): Promise<Student | null> {
    return this.studentModel
      .findOneAndUpdate({ studentId }, updateData, { new: true })
      .exec();
  }

  async incrementAttendance(studentId: string): Promise<Student | null> {
    return this.studentModel
      .findOneAndUpdate(
        { studentId },
        {
          $inc: { totalAttendanceCount: 1 },
          lastAttendanceDate: new Date(),
        },
        { new: true },
      )
      .exec();
  }
}
